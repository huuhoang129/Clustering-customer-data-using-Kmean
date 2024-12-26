import tkinter as tk
import os
from tkinter import ttk 
from tab_1 import upload_and_display_file
from tab_2 import process_csv
from tab_3 import kmean
from tab_4 import create_help_tab

current_dir = os.path.dirname(os.path.abspath(__file__))
icon_path = os.path.join(current_dir, "img", "Logo.ico")

# Giao diện            
window = tk.Tk()
window.title("Phân cụm dữ liệu khách hàng bằng thuật toán Kmean")
window.iconbitmap(icon_path)
window.geometry("1280x650")
window.resizable(width=False, height=False)
notebook = ttk.Notebook(window)

# Tab 1
upload_tab = ttk.Frame(notebook)
notebook.add(upload_tab, text='Tải lên & Hiển thị')
upload_and_display_file(upload_tab)

# Tab 2
handle_tab = ttk.Frame(notebook)
notebook.add(handle_tab, text='Tiền xử lý')
process_csv(handle_tab, upload_tab)


# Tab 3
kmean_tab = ttk.Frame(notebook)
notebook.add(kmean_tab, text="Kmean")
kmean(kmean_tab)


help_tab = ttk.Frame(notebook)
notebook.add(help_tab, text="Hướng dẫn")
create_help_tab(help_tab)

# Chạy file
notebook.pack(fill='both', expand=True)
window.mainloop()
