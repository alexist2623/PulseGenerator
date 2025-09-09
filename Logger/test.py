from email.message import EmailMessage
import smtplib, ssl

username = "qstldots@gmail.com"
password = "tvlx gumn yhir qxmy"
host = "smtp.gmail.com"
port = 587

server = smtplib.SMTP(host, port, timeout=30)
context = ssl.create_default_context()
server.ehlo()
server.starttls(context=context)
server.login(username, password)
