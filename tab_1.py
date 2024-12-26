import tkinter as tk
import pandas as pd

from tkinter import ttk
from tkinter import filedialog

# Hàm tải lên và hiển thị file
def upload_and_display_file(frame):
    frame.df = None  # Khởi tạo dataframe trống 

    # Hàm tải lên dữ liệu từ tệp 
    def handle_upload():
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv"), ("Excel Files", "*.xlsx;*.xls"), ("Text Files", "*.txt")])
        if file_path:
            convert_to_csv_and_display(frame, file_path)

    # Hàm chuyển đổi tệp và hiển thị
    def convert_to_csv_and_display(frame, file_path):
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.txt'):
                df = pd.read_csv(file_path, sep='\t' if file_path.endswith('.tsv') else None, engine='python')
            else:
                raise ValueError("Lỗi file")
            frame.df = df
            
            tree.delete(*tree.get_children())
            # Hiển thị dataframe 
            tree["column"] = list(df.columns)
            tree["show"] = "headings"
            for column in tree["columns"]:
                tree.heading(column, text=column, anchor="center") 
                tree.column(column, anchor="e")
            for index, row in df.iterrows():
                values = list(row)
                for i in range(len(values)):
                    values[i] = str(values[i]).rjust(10)
                tree.insert("", index, values=values)
            label.config(text="{}".format(file_path))
        except Exception as e:
            label.config(text="Đọc file bị lỗi: {}".format(e))

    # Tạo giao diện
# Phần 1: hiển thị nội dung tệp
    frame_P1 = ttk.Frame(frame, borderwidth=3, relief="ridge", width=1220, height=500)
    frame_P1.pack(side='top', pady=10)
    frame_P1.pack_propagate(False)

    frame_tree = ttk.Frame(frame_P1, borderwidth=3, relief="ridge", width=1220, height=476)
    frame_tree.pack(side='top')
    frame_tree.pack_propagate(False)


    tree_frame = ttk.Frame(frame_tree, borderwidth=0, relief="flat", width=1189, height=476)
    tree_frame.pack(side="left", expand=True)
    tree_frame.pack_propagate(False)

    tree = ttk.Treeview(tree_frame)
    tree.pack(side="top", fill="both", expand=True)

    scrollbar_y = tk.Scrollbar(frame_tree, orient="vertical", command=tree.yview)
    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x = tk.Scrollbar(frame_P1, orient="horizontal", command=tree.xview)
    scrollbar_x.pack(side="top", fill="x")
    tree.configure(yscrollcommand=scrollbar_y.set)
    tree.configure(xscrollcommand=scrollbar_x.set)

# Phần 2: hiển thị tên tệp và các nút điều khiển
    frame_P2 = ttk.Frame(frame, borderwidth=0, relief="flat", width=1220, height=100)
    frame_P2.pack(side='top', pady=5, ipady=20)
    frame_P2.pack_propagate(False)
    label = ttk.Label(frame_P2, text="Hãy chọn các file có đuôi .csv, .xsls ,.txt ")
    label.pack(side='left', padx=50)
    frame_upload_clear = ttk.Frame(frame_P2, borderwidth=0, relief="flat", width=200, height=100)
    frame_upload_clear.pack(side='right', padx=20)
    frame_upload_clear.pack_propagate(False)
    upload_button = ttk.Button(frame_upload_clear, text="Upload", command=handle_upload)
    upload_button.pack(side='right',expand=True, ipadx=58, ipady=10, anchor='center')
