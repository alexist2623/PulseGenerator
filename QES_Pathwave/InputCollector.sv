`timescale 1ns / 1ps
//*****************************************************************************
//     ________
//    /  ____  \
//   /  /  __\__\__      Vendor             : SNU QuIQCL
//  |  |  /  ____  \     Version            : 1.0
//   \  \/  /__  \  \    Application        : Collect Input Data
//    \_|  |_\ \__|  |   Filename           : InputCollector.sv
//       \  \_\_\ \ /    Date Last Modified : 2023/08/24 16:55:51
//        \______\_\     Date Created       : 2023/08/24 16:55:51
//
// (c) Copyright 2017 - 2024 QuIQCL. All rights reserved.
//
// Engineer: SNU ECE Jeonghyun Park alexist@snu.ac.kr
// Module Name: InputCollector
// Project Name: lolenc
// Target Devices: Ultrascale+ RFSoC
// Tool Versions: 2020.2
// 
// Dependencies: None
// 
// Revision:
// Revision 0.01 - File Created
//
// Additional Comments:
// 
//*****************************************************************************

module InputCollector
#(
    parameter FIFO_DEPTH         = 512,
    localparam FIFO_DEPTH_WIDTH  = $clog2(FIFO_DEPTH)
)
(
    input  wire        clk,
    input  wire        clk_x4,
    input  wire        datain_i,
    input  wire        reset,
    input  wire [63:0] cmd_in,
    input  wire [63:0] counter,
    input  wire        valid,
    input  wire        delay_en,
    input  wire        delay_value,
    input  wire        delay_load,

    output reg         write,
    output reg  [127:0] count_out,
    output wire [3:0]  curr_mode
);

//
// Data Input IDELAY
//
wire [7:0]  iserdese_out;
wire        fifo_empty;

assign curr_mode = mode;
/*
 * "ug571" page 157
 * When using the FIFO, the FIFO_RD_EN should be driven by the
 * inverted FIFO_EMPTY signal to ensure the FIFO_write and read 
 * pointers do not overlap every eight clock cycles.
 *
 * for data output iserdese_out[7:0], input corresponds to
 * [0] [1] [2] ... [7] in timeline
 */
ISERDESE3 #(
   .DATA_WIDTH          (8),
   .SIM_DEVICE   	    ("ULTRASCALE_PLUS"),
   .FIFO_ENABLE         ("TRUE"),
   .FIFO_SYNC_MODE      ("FALSE") 
)
iserdes_m (
   .D                   (datain_i),
   .RST                 (reset),
   .CLK                 (clk_x4),
   .CLK_B               (~clk_x4),
   .CLKDIV              ( clk),
   .Q                   (iserdese_out), 
   .FIFO_RD_CLK         (clk),
   .FIFO_RD_EN          (~fifo_empty),
   .FIFO_EMPTY          (fifo_empty),
   .INTERNAL_DIVCLK     () // Reserved port
);
/*
 * [ 3: 0]  : mode
 * [63: 4]  : config_data
 */
localparam MODE_NUM        = 5;
localparam MODE_WIDTH      = 4;
localparam PIPELINE_DEPTH  = 2;

localparam IDLE            = MODE_WIDTH'(0);
localparam EDGE_COUNT      = MODE_WIDTH'(1);
localparam TIME_DETECT     = MODE_WIDTH'(2);
localparam FIRST_DETECT    = MODE_WIDTH'(3);
localparam LAST_DETECT     = MODE_WIDTH'(4);
localparam MID_DETECT      = MODE_WIDTH'(5);

//////////////////////////////////////////////////////////
// Pipeline variables
//////////////////////////////////////////////////////////
reg [MODE_WIDTH-1:0]       mode;

reg [127:0]                input_buffer;
reg [3:0]                  input_index; // 128 // 8 = 16, $clog2(16) = 4
reg [FIFO_DEPTH_WIDTH-1:0] fifo_depth;

reg [63 - MODE_WIDTH:0]    config_data;
reg                        count_end;
reg [63:0]                 count_offset;
reg [127:0]                count_mid;

reg [7:0]                  iserdese_out_buffer;
reg                        prev_input;
reg [7:0]                  edge_detect;
reg [3:0]                  edge_sum_0_3[3:0]; 
reg [3:0]                  edge_sum_4_7[3:0]; 
reg [127:0]                edge_num;

always_ff @(posedge clk) begin
    if (reset == 1'b1) begin
        input_index         <= 8'h0;
        input_buffer        <= 128'h0;
        mode                <= IDLE;
        fifo_depth          <= FIFO_DEPTH_WIDTH'(0);
        count_out           <= 128'h0;
        write               <= 1'b0;
        config_data         <= (63 - MODE_WIDTH)'(0);
        edge_num            <= 128'h0;
        count_end           <= 1'b0;
        count_offset        <= 64'h0;
        count_mid           <= 128'h0;
        
        prev_input          <= 1'b0;
        iserdese_out_buffer <= 8'h0;
        edge_detect         <= 8'h0;
        for (int i = 0; i < 3; i++) begin : edge_sum
            edge_sum_0_3[i] <= 4'h0;
            edge_sum_4_7[i] <= 4'h0;
        end
    end
    else begin
        write                <= 1'b0;
        prev_input           <= iserdese_out[7];
        iserdese_out_buffer  <= iserdese_out;

        edge_detect[0]       <= ((~prev_input) & iserdese_out[0]);
        edge_detect[1]       <= ((~iserdese_out[0]) & iserdese_out[1]);
        edge_detect[2]       <= ((~iserdese_out[1]) & iserdese_out[2]);
        edge_detect[3]       <= ((~iserdese_out[2]) & iserdese_out[3]);
        edge_detect[4]       <= ((~iserdese_out[3]) & iserdese_out[4]);
        edge_detect[5]       <= ((~iserdese_out[4]) & iserdese_out[5]);
        edge_detect[6]       <= ((~iserdese_out[5]) & iserdese_out[6]);
        edge_detect[7]       <= ((~iserdese_out[6]) & iserdese_out[7]);

        edge_sum_0_3[0]      <= ((~prev_input) & iserdese_out[0]);
        edge_sum_0_3[1]      <= (
            ((~prev_input) & iserdese_out[0]) + 
            ((~iserdese_out[0]) & iserdese_out[1])
        );
        edge_sum_0_3[2]      <= (
            ((~prev_input) & iserdese_out[0]) + 
            ((~iserdese_out[0]) & iserdese_out[1]) + 
            ((~iserdese_out[1]) & iserdese_out[2])
        );
        edge_sum_0_3[3]      <= (
            ((~prev_input) & iserdese_out[0]) + 
            ((~iserdese_out[0]) & iserdese_out[1]) + 
            ((~iserdese_out[1]) & iserdese_out[2]) + 
            ((~iserdese_out[2]) & iserdese_out[3])
        );

        edge_sum_4_7[0]      <= ((~iserdese_out[3]) & iserdese_out[4]);
        edge_sum_4_7[1]      <= (
            ((~iserdese_out[3]) & iserdese_out[4]) + 
            ((~iserdese_out[4]) & iserdese_out[5])
        );
        edge_sum_4_7[2]      <= (
            ((~iserdese_out[3]) & iserdese_out[4]) + 
            ((~iserdese_out[4]) & iserdese_out[5]) + 
            ((~iserdese_out[5]) & iserdese_out[6])
        );
        edge_sum_4_7[3]      <= (
            ((~iserdese_out[3]) & iserdese_out[4]) + 
            ((~iserdese_out[4]) & iserdese_out[5]) + 
            ((~iserdese_out[5]) & iserdese_out[6]) + 
            ((~iserdese_out[6]) & iserdese_out[7])
        );
        case (mode)        
            IDLE: begin
                count_out   <= 128'h0;
                fifo_depth  <= FIFO_DEPTH_WIDTH'(0);
                edge_num    <= 128'h0;
                count_end   <= 1'b0;
                count_mid   <= 128'h0;
                if (valid == 1'b1) begin
                    mode         <= cmd_in[MODE_WIDTH - 1:0];
                    config_data  <= cmd_in[63:MODE_WIDTH];
                    count_offset <= counter;
                end
            end
            EDGE_COUNT: begin
                /*
                * Count the number of edges in the input data
                */
                if (edge_num[63:0] < 32'hffff_fff0) begin
                    edge_num[63:0] <= (
                        edge_num[63:0] + 
                        edge_sum_0_3[3] +
                        edge_sum_4_7[3]
                    );
                end
                if (valid == 1'b1) begin
                    write        <= 1'b1;
                    count_out    <= {64'h0, edge_num[63:0]};
                    fifo_depth   <= FIFO_DEPTH_WIDTH'(0);
                    mode         <= cmd_in[MODE_WIDTH - 1:0];
                    config_data  <= cmd_in[63:MODE_WIDTH];
                    edge_num     <= 128'h0;
                    count_end    <= 1'b0;
                    count_offset <= counter;
                    count_mid    <= 128'h0;
                end
            end
            TIME_DETECT: begin
                if (fifo_depth < FIFO_DEPTH) begin
                    write                              <= 1'b0;
                    input_buffer[input_index * 8 +: 8] <= iserdese_out[7:0];
                    if (input_index == 4'hf) begin
                        input_index    <= 8'h0;
                        input_buffer   <= 128'h0;
                        count_out      <= input_buffer;
                        fifo_depth     <= fifo_depth + 1;
                        write          <= 1'b1;
                    end
                    else begin
                        input_index    <= input_index + 1;
                    end
                end
                if (valid == 1'b1) begin
                    write        <= 1'b1;
                    count_out    <= input_buffer;
                    input_buffer <= 128'h0;
                    fifo_depth   <= FIFO_DEPTH_WIDTH'(0);
                    mode         <= cmd_in[MODE_WIDTH - 1:0];
                    config_data  <= cmd_in[63:MODE_WIDTH];
                    edge_num     <= 128'h0;
                    count_end    <= 1'b0;
                    count_offset <= counter;
                    count_mid    <= 128'h0;
                end
            end
            FIRST_DETECT: begin
                if (count_end == 1'b0 && |iserdese_out_buffer == 1'b1) begin
                count_end <= 1'b1;
                if (edge_detect[0] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h0};
                end
                else if (edge_detect[1] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h1};
                end
                else if (edge_detect[2] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h2};
                end
                else if (edge_detect[3] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h3};
                end
                else if (edge_detect[4] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h4};
                end
                else if (edge_detect[5] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h5};
                end
                else if (edge_detect[6] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h6};
                end
                else if (edge_detect[7] == 1'b1) begin
                        edge_num <= {(counter[60:0] - count_offset[60:0]), 3'h7};
                end
                end
                if (valid == 1'b1) begin
                    write        <= 1'b1;
                    count_out    <= edge_num;
                    fifo_depth   <= FIFO_DEPTH_WIDTH'(0);
                    mode         <= cmd_in[MODE_WIDTH - 1:0];
                    config_data  <= cmd_in[63:MODE_WIDTH];
                    edge_num     <= 128'h0;
                    count_end    <= 1'b0;
                    count_offset <= counter;
                    count_mid    <= 128'h0;
                end
            end
            LAST_DETECT: begin
                if( edge_detect[7] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h7};
                end
                else if( edge_detect[6] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h6};
                end
                else if( edge_detect[5] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h5};
                end
                else if( edge_detect[4] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h4};
                end
                else if( edge_detect[3] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h3};
                end
                else if( edge_detect[2] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h2};
                end
                else if( edge_detect[1] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h1};
                end
                else if( edge_detect[0] == 1'b1 ) begin
                    edge_num <= {(counter[60:0]-count_offset[60:0]),3'h0};
                end

                if( valid == 1'b1 ) begin
                    write           <= 1'b1;
                    count_out       <= edge_num;
                    fifo_depth      <= FIFO_DEPTH_WIDTH'(0);
                    mode            <= cmd_in[MODE_WIDTH - 1:0];
                    config_data     <= cmd_in[63:MODE_WIDTH];
                    edge_num        <= 128'h0;
                    count_end       <= 1'b0;
                    count_offset    <= counter;
                    count_mid       <= 128'h0;
                end
            end
            MID_DETECT: begin
                if (count_end == 1'b0) begin
                    if (edge_num[31:0] < 32'hffff_fff0) begin
                        edge_num[31:0] <= (edge_num[31:0] + edge_sum_0_3[3] + edge_sum_4_7[3]);
                    end
                    if ((edge_num[31:0] + edge_sum_0_3[0]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h0};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0]+edge_sum_0_3[1]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h1};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0] + edge_sum_0_3[2]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h2};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0] + edge_sum_0_3[3]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h3};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0] + edge_sum_0_3[3] + edge_sum_4_7[0]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h4};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0] + edge_sum_0_3[3] + edge_sum_4_7[1] ) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h5};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0] + edge_sum_0_3[3] + edge_sum_4_7[2]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h6};
                        count_end <= 1'b1;
                    end
                    else if ((edge_num[31:0] + edge_sum_0_3[3] + edge_sum_4_7[3]) == config_data[31:0]) begin
                        count_mid <= {(counter[60:0]-count_offset[60:0]),3'h7};
                        count_end <= 1'b1;
                    end
                end
                if( valid == 1'b1 ) begin
                    write           <= 1'b1;
                    count_out       <= count_mid;
                    fifo_depth      <= FIFO_DEPTH_WIDTH'(0);
                    mode            <= cmd_in[MODE_WIDTH - 1:0];
                    config_data     <= cmd_in[63:MODE_WIDTH];
                    edge_num        <= 128'h0;
                    count_end       <= 1'b0;
                    count_offset    <= counter;
                    count_mid       <= 128'h0;
                end
            end
        endcase
    end
end

endmodule