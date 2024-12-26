import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import pandas as pd
import re
import unicodedata
import numpy as np
from tkinter import filedialog
from sklearn.impute import KNNImputer


prep_df = None
def process_csv(frame, upload_display_frame):
    # Hàm hiển thị thông tin dữ liệu
    def preprocess():
        if hasattr(upload_display_frame, 'df') and upload_display_frame.df is not None:
            df = upload_display_frame.df
            num_rows = len(df)
            num_empty_cells = df.isnull().sum().sum()
            num_columns = df.shape[1]
            column_names = ", ".join(df.columns.tolist())
            empty_cells_per_column = df.isnull().sum()

            column_info = ""
            for column, empty_cells in empty_cells_per_column.items():
                column_info += f" - Trường {column}: {empty_cells} ô thiếu\n"

            text_columns = []
            numeric_columns = []

            for column, empty_cells in empty_cells_per_column.items():
                if df[column].dtype == 'object':
                    text_columns.append(column)
                else:
                    numeric_columns.append(column)

            text_column_names = ", ".join(text_columns)
            numeric_column_names = ", ".join(numeric_columns)
            
            # Tìm Trường chứa nhiều dữ liệu trùng lặp và ít dữ liệu trùng lặp
            duplicate_info = ""
            kmeans_warning = ""
            for column in df.columns:
                unique_count = df[column].nunique()
                duplicate_count = num_rows - unique_count
                duplicate_ratio = duplicate_count / num_rows
                if unique_count < num_rows * 0.1:
                    duplicate_info += f" - Trường {column} có nhiều dữ liệu trùng lặp (chỉ có {unique_count} giá trị duy nhất)\n"
                    # Kiểm tra nếu trường chứa nhiều dữ liệu số trùng lặp hơn 50 giá trị
                    if df[column].dtype in ['int64', 'float64']:
                        if duplicate_ratio > 0.3:
                            kmeans_warning += f" - Trường {column} có nhiều dữ liệu số trùng lặp (> 30%), ảnh hưởng nặng đến thuật toán K-means\n"
                        elif duplicate_ratio > 0.1:
                            kmeans_warning += f" - Trường {column} có nhiều dữ liệu số trùng lặp (10% - 30%), ảnh hưởng trung bình đến thuật toán K-means\n"
                        else:
                            kmeans_warning += f" - Trường {column} có nhiều dữ liệu số trùng lặp (< 10%), ảnh hưởng nhẹ đến thuật toán K-means\n"
                elif unique_count > num_rows * 0.9:
                    duplicate_info += f" - Trường {column} có ít dữ liệu trùng lặp (có {unique_count} giá trị duy nhất)\n"


            # Tìm trường có chứa dữ liệu âm
            negative_info = ""
            for column in df.select_dtypes(include=['number']).columns:
                if (df[column] < 0).any():
                    negative_info += f" - Trường {column} có chứa dữ liệu âm\n"

            info_text = "Tổng số dòng trong file là: {}\nTổng số ô trống trong file là: {}\nTổng số trường trong file là: {}\nChi tiết Trường là: {}\n".format(num_rows, num_empty_cells, num_columns, column_names)
            info_text += "\nTrường chứa dữ liệu chữ và số là:"
            info_text += "\n- Trường chứa nhiều dữ liệu là chữ là: {}\n".format(text_column_names)
            info_text += "- Trường chứa nhiều dữ liệu là số là: {}\n".format(numeric_column_names)

            info_text += "\nSố ô trống cho mỗi trường là:\n"
            info_text += column_info
            info_text += "\nThông tin dữ liệu trùng lặp là:\n"
            info_text += duplicate_info
            info_text += "\nThông tin dữ liệu âm là:\n"
            info_text += negative_info
            info_text += "\nCảnh báo:\n"
            info_text += kmeans_warning

            info_text_widget.config(state=tk.NORMAL)
            info_text_widget.delete('1.0', tk.END)
            info_text_widget.insert(tk.END, info_text)
            info_text_widget.config(state=tk.DISABLED)
        else:
            info_text_widget.config(state=tk.NORMAL)
            info_text_widget.delete('1.0', tk.END)
            info_text_widget.insert(tk.END, "Vui lòng tải lên một tệp CSV trước khi tiền xử lý.")
            info_text_widget.config(state=tk.DISABLED)

        # Hàm xử lý dữ liệu
    def prep():
        global prep_df
        if hasattr(upload_display_frame, 'df') and upload_display_frame.df is not None:  
            df = upload_display_frame.df

        # Làm sạch dữ liệu:
            # Loại bỏ những dòng trùng lặp
            def remove_duplicates(df):
                df.drop_duplicates(inplace=True)
                return df
            
            # Hàm xóa dòng thiếu quá nhiều dữ liệu
            def delete_missing_line(df):
                quantity = df.isnull().sum(axis=1)
                df_cleaned = df[quantity < 3]
                return df_cleaned

            # Chuẩn hóa văn bản và sửa lôi chính tả
            def process_dataframe(df):
                processed_df = df
                def process_text(text):
                    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
                    text = text.lower()
                    text = re.sub(r'[^a-z0-9\s]', '', text)
                    text = re.sub(r'\s', '', text)
                    return text
                for column in processed_df.columns:
                    processed_df[column] = processed_df[column].apply(lambda x: process_text(x) if isinstance(x, str) else x)
                return processed_df
            
            # Hàm mã hóa giới tính
            def gender_coding(df):
                for column in df.columns:
                    gender_keywords = ['male', 'female', 'nam', 'nữ', 'm', 'f', 'mister', 'miss', '_', '/']
                    if df[column].apply(lambda x: str(x).lower() in gender_keywords).any():
                        mapping_dict = {'male': 0, 'female': 1, 'nam': 0, 'nữ': 1, 'm': 0, 'f': 1, 'mister': 0, 'miss': 1, '_': 0, '/': 1}
                        df[column] = df[column].map(mapping_dict)
                return df
            
            # Hàm loại bỏ Trường chứa các dữ liệu dài như địa chỉ, họ tên
            def delete_columns_with_many_text_data(df, threshold=50):
                columns_to_drop = [] 
                for column in df.columns:  
                    if pd.api.types.is_string_dtype(df[column]):
                        num_text_data = df[column].apply(lambda x: isinstance(x, str)).sum()
                        if num_text_data >= threshold:
                            columns_to_drop.append(column)
                df.drop(columns=columns_to_drop, inplace=True) 
                return df
            
            # Hàm loại bỏ trường dư thừa
            def drop_id_column(df):
                consecutive_increase_threshold = 100 
                for col in df.columns:
                    consecutive_count = 0
                    prev_value = None
                    for value in df[col]:
                        if prev_value is not None and value == prev_value + 1:
                            consecutive_count += 1
                            if consecutive_count >= consecutive_increase_threshold:
                                df = df.drop(columns=[col])
                                break
                        else:
                            consecutive_count = 0
                        prev_value = value
                return df
            
            
            # Tiền xử lý dữ liệu
            # Hàm xử lý dữ liệu âm
            def handle_negative_data(df):
                for column in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        df[column] = df[column].apply(lambda x: abs(x) if x < 0 else x)
                    else:
                        df[column] = df[column].str.replace(r'^-', '', regex=True)
                return df
            
            # Hàm xử lý dữ liệu giới tính bị thiếu
            def fill_gender_na(df):
                for column in df.columns:
                    if df[column].isin([0, 1]).sum() >= 1:
                        gender_column = column
                        break
                else:
                    return df
                p_female = df[gender_column].sum() / len(df[gender_column])
                p_male = 1 - p_female
                
                # Điền các ô trống trong trường 'Gender' dựa trên phân phối Bernoulli
                for index, row in df.iterrows():
                    if pd.isnull(row[gender_column]):
                        df.at[index, gender_column] = np.random.choice([0, 1], p=[p_male, p_female])
                return df
            
            def fill_missing_values(df, n_neighbors=5):
                imputer = KNNImputer(n_neighbors=n_neighbors)            
                data_imputed = imputer.fit_transform(df)               
                data_filled = pd.DataFrame(data_imputed, columns=df.columns)
                data_filled = np.floor(data_filled).astype(int)
                return data_filled
                
            # Làm sạch dữ liệu
            df = delete_missing_line(df)
            df = process_dataframe(df)
            df = gender_coding(df)
            df = remove_duplicates(df)
            df = delete_columns_with_many_text_data(df)
            df = drop_id_column(df)

            # Tiền xử lý dữ liệu
            df = handle_negative_data(df)
            df = fill_gender_na(df)
            df = fill_missing_values(df)

            global prep_df  
            prep_df = df  


            # Xóa dữ liệu hiện tại trong treeview
            for item in tree.get_children():
                tree.delete(item)
            tree["column"] = list(df.columns)
            tree["show"] = "headings"
            for column in tree["column"]:
                tree.heading(column, text=column)
            for index, row in df.iterrows():
                tree.insert("", "end", values=list(row))
        else:
            pass

    # Hàm clear text
    def clear_text():
        info_text_widget.config(state=tk.NORMAL)
        info_text_widget.delete("1.0", tk.END)
        info_text_widget.config(state=tk.DISABLED)
        for widget in tree.get_children():
            tree.delete(widget)

     # Hàm xuất file csv
    def export_csv():
        try:
            if prep_df is not None:
                file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
                if file_path:
                    prep_df.to_csv(file_path, index=False)
                    label_save_file.config(text=f"Dữ liệu đã được lưu thành công vào {file_path}")
            else:
                label_save_file.config(text="Không có dữ liệu đã tiền xử lý để xuất. Vui lòng tiền xử lý trước khi xuất văn bản.")
        except Exception as e:
            label_save_file.config(text=f"Đã xảy ra lỗi khi lưu tệp: {str(e)}")

    def export_excel():
        try:
            if prep_df is not None:
                file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
                if file_path:
                    prep_df.to_excel(file_path, index=False)
                    label_save_file.config(text=f"Dữ liệu đã được lưu thành công vào {file_path}")
            else:
                label_save_file.config(text="Không có dữ liệu đã tiền xử lý để xuất. Vui lòng tiền xử lý trước khi xuất văn bản.")
        except Exception as e:
            label_save_file.config(text=f"Đã xảy ra lỗi khi lưu tệp: {str(e)}")

    def export_txt():
        try:
            if prep_df is not None:
                file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
                if file_path:
                    prep_df.to_csv(file_path, index=False, sep=',')
                    label_save_file.config(text=f"Dữ liệu đã được lưu thành công vào {file_path}")
            else:
                label_save_file.config(text="Không có dữ liệu đã tiền xử lý để xuất. Vui lòng tiền xử lý trước khi xuất văn bản.")
        except Exception as e:
            label_save_file.config(text=f"Đã xảy ra lỗi khi lưu tệp: {str(e)}")


    # Tạo giao diện
    # Phần 1: Hiển thị thông tin file và chức năng
    frame_P1 = ttk.Frame(frame, borderwidth=1, relief="solid", width=617, height=625)
    frame_P1.pack(side='left', expand=True)
    frame_P1.pack_propagate(False)

    # Khung label và button
    frame_label_and_button = ttk.Frame(frame_P1, borderwidth=1, relief="solid", width=613, height=203)
    frame_label_and_button.pack(side='top', pady=1, padx=1)

         # 1. Khung Label
    frame_label_function = ttk.Frame(frame_label_and_button, borderwidth=0, relief="solid", width=612, height=50)
    frame_label_function.pack(side='top')
    frame_label_function.pack_propagate(False)

    label_function = ttk.Label(frame_label_function, text='Chức Năng Chính', font=('Helvetica', 13, 'bold'))
    label_function.pack(side='left', padx=5, pady=10)

    gifImage = "img/decorate_duck.gif"
    openImage = Image.open(gifImage)

    new_width = 50
    new_height = 55
    imageObject = []
    for frame_num in range(openImage.n_frames):
        openImage.seek(frame_num)
        resized_frame = openImage.resize((new_width, new_height), Image.Resampling.LANCZOS)
        imageObject.append(ImageTk.PhotoImage(resized_frame))
    count = 0

    def animation(count):
        newImage = imageObject[count]
        gif_Label.configure(image=newImage)
        count += 1
        if count == openImage.n_frames:
            count = 0
        frame_label_function.after(50, lambda: animation(count))

    gif_Label = ttk.Label(frame_label_function, image="")
    gif_Label.pack(side='left', padx=1, pady=1)
    animation(count)

        # 2. Khung Button
    frame_button_info_and_txl = ttk.Frame(frame_label_and_button, borderwidth=0, relief="solid", width=612, height=50)
    frame_button_info_and_txl.pack(side='top', ipadx=22)
    frame_button_info_and_txl.pack_propagate(False)

    preprocess_button = ttk.Button(frame_button_info_and_txl, text="Thông tin dữ liệu", command=preprocess)
    preprocess_button.pack(side='left', ipadx=62, ipady=10, fill='both', expand=True)
    prep_button = ttk.Button(frame_button_info_and_txl, text="Tiền xử lý", command=prep)
    prep_button.pack(side='left', ipadx=62, ipady=10, fill='both', expand=True)
    clear_button = ttk.Button(frame_button_info_and_txl, text="Xóa toàn bộ", command=clear_text)
    clear_button.pack(side='left', ipadx=62, ipady=10, fill='both', expand=True)

        # Khung label and text
    frame_label_and_text = ttk.Frame(frame_P1, borderwidth=1, relief="solid", width=613, height=516)
    frame_label_and_text.pack(side='top', pady=5)
    frame_label_and_text.pack_propagate(False)

    frame_label_info_file = ttk.Frame(frame_label_and_text, borderwidth=0, relief="solid", width=612, height=50)
    frame_label_info_file.pack(side='top', pady=1, padx=1)
    frame_label_info_file.pack_propagate(False)

    info_csv = ttk.Label(frame_label_info_file, text="Thông Tin Dữ Liệu", font=('Helvetica', 13, 'bold'))
    info_csv.pack(side='left', pady=6)

    info_text_widget = tk.Text(frame_label_and_text, wrap="word")
    info_text_widget.pack(side='top', ipady=32)

    # Phần 2: Hiển thi file đã xử lý
    frame_P2 = ttk.Frame(frame, borderwidth=0, relief="solid", width=617, height=625)
    frame_P2.pack(side='left', expand=True)
    frame_P2.pack_propagate(False)

    frame_tree = ttk.Frame(frame_P2, borderwidth=3, relief="ridge", width=617, height=500)
    frame_tree.pack(side='top')
    frame_tree.pack_propagate(False)

        # Khung text and y
    frame_text_and_y_2 = ttk.Frame(frame_tree, borderwidth=0, relief="solid", width=610, height=474)
    frame_text_and_y_2.pack(side='top', pady=1)
    frame_text_and_y_2.pack_propagate(False)

    tree_frame = ttk.Frame(frame_text_and_y_2, borderwidth=0, relief="solid", width=590, height=500)
    tree_frame.pack(side="left", expand=True)
    tree_frame.pack_propagate(False)

            # Thanh y kèm text 2
    tree = ttk.Treeview(tree_frame)
    tree.pack(side="top", fill="both", expand=True)

    scrollbar_y = tk.Scrollbar(frame_text_and_y_2, orient="vertical", command=tree.yview)
    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x = tk.Scrollbar(frame_tree, orient="horizontal", command=tree.xview)
    scrollbar_x.pack(side="top", fill="x")

    tree.configure(yscrollcommand=scrollbar_y.set)
    tree.configure(xscrollcommand=scrollbar_x.set)

    # Khung chữ và nút
    frame_label_and_button_2 = ttk.Frame(frame_P2, borderwidth=0, relief="solid", width=612, height=120)
    frame_label_and_button_2.pack(side='top',pady=2)
    frame_label_and_button_2.pack_propagate(False)

    label_save_file = ttk.Label(frame_label_and_button_2, text="Chọn dạng file bạn muốn lưu:", font=("Arial", 10))
    label_save_file.pack(side='left', padx=10, expand=True)

    frame_button_export = ttk.Frame(frame_label_and_button_2, borderwidth=0, relief="solid", width=50, height=30)
    frame_button_export.pack(side='right', padx= 50, ipadx=50, ipady=35, pady=3)
    frame_button_export.pack_propagate(False)

    export_button_csv = ttk.Button(frame_button_export, text="CSV", command=export_csv)
    export_button_csv.pack(side='top', ipadx=12,ipady=3, fill='both', expand=True)

    export_button_excel = ttk.Button(frame_button_export, text="Excel", command=export_excel)
    export_button_excel.pack(side='top', ipadx=12,ipady=3, fill='both', expand=True)

    export_button_txt = ttk.Button(frame_button_export, text="Txt", command=export_txt)
    export_button_txt.pack(side='top', ipadx=12,ipady=3,fill='both', expand=True)