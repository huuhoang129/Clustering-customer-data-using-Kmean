import csv

def csv_to_txt(csv_file, txt_file):
  with open(csv_file, 'r') as csvfile, open(txt_file, 'w') as txtfile:
    reader = csv.reader(csvfile)
    for row in reader:
      data = ','.join(row).replace('"', '')
      txtfile.write(data + '\n')

# Ví dụ sử dụng
csv_file = 'file/mail.csv'
txt_file = 'data.txt'
csv_to_txt(csv_file, txt_file)
