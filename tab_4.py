import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage

def create_help_tab(frame):

    normal_font = ('Helvetica', 10)
    bold_font = ('Helvetica', 15, 'bold')

    frame_P1 = ttk.Frame(frame, borderwidth=3, relief="ridge", width=1300, height=720)
    frame_P1.pack(side='top', pady=5)
    frame_P1.pack_propagate(False)

    help_gioithieu_o = ttk.Label(frame_P1, text="\nGIỚI THIỆU", font=bold_font)
    help_gioithieu_o.pack(padx=10, pady=5, anchor='w')

    help_gt = (
        "Ứng dụng này được thiết kế để giúp bạn phân khúc khách hàng dựa trên thuật toán K-means. "
        "Các bước thực hiện như sau:\n"
        "    1. Tải lên tệp tin có chứa dữ liệu khách hàng.\n"
        "    2. Tiền xử lý dữ liệu để chuẩn hóa dữ liệu.\n"
        "    3. Chọn số cụm và chạy thuật toán K-means.\n"
        "    4. Xem kết quả phân cụm.\n"
        "    5. Hướng dẫn sử dụng\n\n"
        "Về ứng dụng sẽ có 4 tab bao gồm:\n"
        "Tab 1: Tải lên và hiển thị dữ liệu\n"
        "  Nhấp vào nút 'Chọn tệp' và chọn các tệp tin bao gồm txt, csv và excel chứa dữ liệu khách hàng của bạn. Sau đó nhấp vào nút 'Tải lên'.\n"
        "    => Dữ liệu của bạn sẽ được tải lên và hiển thị trong bảng.\n"
        "Tab 2: Tiền xử lý \n"
        "  Ở phần này khi ấn nút 'Thông tin' sẽ hiện ra thông tin của file trước khi tiền xử lý \n"
        "  Nhấp vào nút 'Tiền xử lý'.\n"
        "    => Sau đó, dữ liệu của bạn sẽ được tiền xử lý và hiển thị trong bảng.\n"
        "   Có thể nhấn nút 'Xóa dữ liệu' để xóa cái dữ liệu trước đó.\n"
        "Tab 3: K-means\n"
        "  Ở phần này sẽ tải lên 2 file gồm file đã tiền xử lý và  file gốc\n"
        "Sau đó nhập số cụm bạn muốn. Chọn các cột dữ liệu bạn muốn sử dụng để phân cụm. Nhấp vào nút 'Chạy Kmeans'.\n"
        "    => Thuật toán K-means sẽ được chạy và kết quả sẽ được hiển thị trong bảng.\n"
        " Nhấp các nút 'Vẽ biểu đồ' để vẽ biểu đồ 2 dạng, chọn cụm và trường để tính toán thống kê liên quan\n"
        "Tab 4: Hướng dẫn\n"
        "    => Xem hướng dẫn sử dụng ứng dụng ở đây"
    )
    help_gioithieu_content = ttk.Label(frame_P1, text=help_gt, font=normal_font)
    help_gioithieu_content.pack(padx=5, pady=1, anchor='w')

    help_ly = ttk.Label(frame_P1, text="LƯU Ý:", font=bold_font)
    help_ly.pack(padx=10, pady=5, anchor='w')

    help_luu_y = (
        "  - Dữ liệu của bạn phải được chuẩn hóa trước khi chạy thuật toán K-means.\n"
        "  - Số lượng cụm bạn chọn sẽ ảnh hưởng đến kết quả phân cụm.\n"
        "  - Chúc bạn thành công!"
    )
    help_luuy_content = ttk.Label(frame_P1, text=help_luu_y, font=normal_font)
    help_luuy_content.pack(padx=10, pady=5, anchor='w')

    help_lienhe = (
        "Liên hệ hỗ trợ:\n"
        "Nếu bạn cần trợ giúp, vui lòng liên hệ với chúng tôi qua email: lehoang12093@gmail.com"
    )
    help_lienhe_content = ttk.Label(frame_P1, text=help_lienhe, font=normal_font)
    help_lienhe_content.pack(padx=10, pady=5, anchor='w')

        # Tải và thêm hình ảnh vào frame
    img = PhotoImage(file="img/HUMG.png")  # Thay đường dẫn bằng đường dẫn thực tế đến ảnh của bạn
    img = img.subsample(7,7)
    img_label = ttk.Label(frame_P1, image=img)
    img_label.image = img  # Giữ một tham chiếu đến ảnh
    img_label.place(x=1050, y=10)