"""Persistent sweep membership independent of table rows and display names."""
from __future__ import annotations

from math import prod
from uuid import uuid4


HARDWARE_GROUP_KINDS = frozenset({
    "amplitude", "ramp_duration", "hold_duration", "rf_duration",
    "rf_frequency", "rf_power",
})

def is_software_sweep(spec):
    # CPMG N can use either executor, so axis_kind alone is insufficient.
    from dc_waveform_core import QickRfSoftwareSweepAxisSpec
    return isinstance(spec, QickRfSoftwareSweepAxisSpec)


def parameter_key(spec):
    return (str(getattr(spec, "axis_kind", "amplitude")),
            str(spec.output_name), str(spec.segment_name))


class SweepGroupingModel:
    """Own stable member identities and one count per hardware sweep group.

    Keys locate the current source parameter; IDs survive source-row remaps.
    Synchronizing after deletion discards its ID, so re-adding the same key
    creates an independent member. Regrouping moves selected members only.
    """

    def __init__(self):
        self.parameters = {}  # source key -> stable UUID
        self.groups = {}      # group UUID -> name/count/member UUIDs
        self.last_counts = {}
        self.execution_order = []  # stable member IDs, slowest to fastest

    def _normalize(self):
        live = set(self.parameters.values())
        for group_id, group in list(self.groups.items()):
            group["members"] = [member for member in group["members"] if member in live]
            if len(group["members"]) < 2:
                del self.groups[group_id]
        self.last_counts = {pid: count for pid, count in self.last_counts.items() if pid in live}
        self.execution_order = [pid for pid in self.execution_order if pid in live]

    def synchronize(self, specs, *, accept_count_edits=True):
        specs = tuple(specs)
        keys = {parameter_key(spec) for spec in specs}
        self.parameters = {key: pid for key, pid in self.parameters.items() if key in keys}
        for spec in specs:
            self.parameters.setdefault(parameter_key(spec), uuid4().hex)
        self._normalize()
        by_id = {self.parameters[parameter_key(spec)]: spec for spec in specs}
        updates = {}
        for group in self.groups.values():
            if accept_count_edits:
                changed = [pid for pid in group["members"]
                           if pid in self.last_counts
                           and int(by_id[pid].count) != self.last_counts[pid]]
                if changed:
                    group["count"] = int(by_id[changed[0]].count)
            for pid in group["members"]:
                spec = by_id[pid]
                if int(spec.count) != group["count"]:
                    updates[parameter_key(spec)] = group["count"]
        self.last_counts = {pid: int(spec.count) for pid, spec in by_id.items()}
        return updates

    def member_id(self, spec):
        return self.parameters.get(parameter_key(spec))

    def group_for(self, spec):
        pid = self.member_id(spec)
        return next(((gid, group) for gid, group in self.groups.items()
                     if pid in group["members"]), (None, None))

    def create_group(self, member_ids, specs):
        specs = tuple(specs)
        by_id = {self.member_id(spec): spec for spec in specs}
        members = list(dict.fromkeys(member_ids))
        if len(members) < 2 or any(pid not in by_id for pid in members):
            raise ValueError("Select at least two existing sweep parameters")
        if any(parameter_key(by_id[pid])[0] not in HARDWARE_GROUP_KINDS for pid in members):
            raise ValueError("Grouping requires hardware sweep parameters; software sweeps remain independent")
        count = int(by_id[members[0]].count)
        for group in self.groups.values():
            group["members"] = [pid for pid in group["members"] if pid not in members]
        self._normalize()
        gid = uuid4().hex
        used_names = {group["name"] for group in self.groups.values()}
        number = 1
        while f"Group {number}" in used_names:
            number += 1
        self.groups[gid] = {"name": f"Group {number}", "count": count, "members": members}
        return gid

    def remove_member(self, member_id):
        for group in self.groups.values():
            group["members"] = [pid for pid in group["members"] if pid != member_id]
        self._normalize()

    def remap(self, key_mapping):
        updated = {}
        for key, pid in self.parameters.items():
            destination = key_mapping.get(key, key)
            if destination is not None:
                destination = tuple(destination)
                if destination in updated:
                    raise ValueError("Sweep parameter remap is not one-to-one")
                updated[destination] = pid
        self.parameters = updated
        self._normalize()

    def loop_specs(self, specs):
        seen, result = set(), []
        for spec in specs:
            group_id, _ = self.group_for(spec)
            identity = ("group", group_id) if group_id else ("parameter", parameter_key(spec))
            if identity not in seen:
                seen.add(identity)
                result.append(spec)
        return tuple(result)

    def ordered_specs(self, specs):
        """Display the execution nesting; correlated members stay together.

        Host recompilation necessarily surrounds hardware loops. Within each
        execution domain, an explicit order overrides the legacy duration-first
        order. Old settings keep that legacy order until a member is moved.
        """
        specs = tuple(specs)
        rank = {pid: i for i, pid in enumerate(self.execution_order)}
        def order(item):
            index, spec = item
            kind = parameter_key(spec)[0]
            host = is_software_sweep(spec)
            legacy = {"ramp_duration": 0, "hold_duration": 1}.get(kind, 2)
            pid = self.member_id(spec)
            return (not host, rank.get(pid, len(rank)), legacy if not host else 0, index)
        ordered = [spec for _, spec in sorted(enumerate(specs), key=order)]
        blocks, seen = [], set()
        for spec in ordered:
            gid, _ = self.group_for(spec)
            key = ("group", gid) if gid else ("parameter", parameter_key(spec))
            if key in seen:
                continue
            seen.add(key)
            blocks.append([item for item in ordered if self.group_for(item)[0] == gid]
                          if gid else [spec])
        return tuple(spec for block in blocks for spec in block)

    def _move_blocks(self, member_id, direction, specs):
        if direction not in (-1, 1):
            raise ValueError("Sweep movement must be -1 or +1")
        ordered = self.ordered_specs(specs)
        blocks = []
        for spec in ordered:
            gid, _ = self.group_for(spec)
            if blocks and gid and self.group_for(blocks[-1][0])[0] == gid:
                blocks[-1].append(spec)
            else:
                blocks.append([spec])
        index = next((i for i, block in enumerate(blocks)
                      if any(self.member_id(spec) == member_id for spec in block)), None)
        if index is None or not 0 <= index + direction < len(blocks):
            return None
        other = index + direction
        host = lambda block: is_software_sweep(block[0])
        if host(blocks[index]) != host(blocks[other]):
            return None  # A host coordinate owns a complete hardware sweep.
        return blocks, index, other

    def can_move(self, member_id, direction, specs):
        return self._move_blocks(member_id, direction, specs) is not None

    def move(self, member_id, direction, specs):
        move = self._move_blocks(member_id, direction, specs)
        if move is None:
            return False
        blocks, index, other = move
        blocks[index], blocks[other] = blocks[other], blocks[index]
        self.execution_order = [self.member_id(spec) for block in blocks for spec in block]
        return True

    def compiler_order(self, specs):
        return tuple((kind, segment, output)
                     for kind, output, segment in map(parameter_key, self.ordered_specs(specs)))

    def point_count(self, specs):
        return prod(int(spec.count) for spec in self.loop_specs(specs))

    def compiler_mapping(self, specs):
        result = {}
        for spec in specs:
            group_id, _ = self.group_for(spec)
            if group_id:
                kind, output, segment = parameter_key(spec)
                result[(kind, segment, output)] = group_id
        return result

    def to_dict(self):
        return {
            "parameters": [{"parameter_id": pid, "key": list(key)}
                           for key, pid in self.parameters.items()],
            "groups": [{"group_id": gid, **group, "members": list(group["members"])}
                       for gid, group in self.groups.items()],
            "execution_order": list(self.execution_order),
        }

    @classmethod
    def from_dict(cls, value):
        model = cls()
        if value is None:
            return model
        if not isinstance(value, dict):
            raise TypeError("sweep_grouping must be an object")
        used_ids = set()
        for entry in value.get("parameters", []):
            key = tuple(entry["key"])
            pid = str(entry["parameter_id"])
            if len(key) != 3 or not all(isinstance(part, str) and part for part in key):
                raise ValueError("Sweep parameter keys require kind/output/segment")
            if not pid or pid in used_ids or key in model.parameters:
                raise ValueError("Sweep parameter identities must be unique")
            model.parameters[key] = pid
            used_ids.add(pid)
        assigned = set()
        order = value.get("execution_order", [])
        if not isinstance(order, list) or any(not isinstance(pid, str) for pid in order):
            raise ValueError("Sweep execution order must be a list of parameter IDs")
        if len(order) != len(set(order)) or any(pid not in used_ids for pid in order):
            raise ValueError("Sweep execution order requires unique existing parameter IDs")
        model.execution_order = list(order)
        for entry in value.get("groups", []):
            gid = str(entry["group_id"])
            count = entry["count"]
            if not gid or gid in model.groups or isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError("Sweep groups require unique IDs and a positive integer count")
            members = list(dict.fromkeys(str(pid) for pid in entry["members"] if str(pid) in used_ids))
            if assigned.intersection(members):
                raise ValueError("A sweep parameter cannot belong to multiple groups")
            if any(key[0] not in HARDWARE_GROUP_KINDS for key, pid in model.parameters.items() if pid in members):
                raise ValueError("Only hardware sweep parameters can belong to a group")
            assigned.update(members)
            model.groups[gid] = {"name": str(entry.get("name", "Group")),
                                 "count": count, "members": members}
        model._normalize()
        return model
