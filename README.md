# Parallel Programing Face Mask Detection

# THÀNH VIÊN:
| No  | Họ Tên  | MSSV |
| :------------ |:---------------:| -----:|
| 1      | Lê Hữu Phúc | 1712667 |
| 2      | Trần Đức Phú |   1712664 |
| 3 | Lê Kinh Luân        | 1612355 |

# KẾ HOẠCH PHÂN CÔNG:
[Link sheets](https://docs.google.com/spreadsheets/d/1aliCbcj5VMrNOlHLznjXiR95PLwh1XnNA3xLeC2Zq_o/edit?usp=sharing)

# MÔ TẢ ỨNG DỤNG:
- Tên ứng dụng: Nhận diện người đeo khẩu trang với mô hình yolov3

- Input: Một tấm ảnh người có/ không có/ đeo sai khẩu trang

- Output: Ảnh được đánh bounding box kèm nhãn mask/ no_mask/ incorrect_mask

- Ý nghĩa thực tế: Khi tình hình dịch đang còn lây lan mạnh trên cả nước thì việc đeo khẩu trang ra đường thực sự là rất cần thiết vì nó không chỉ bảo vệ bản thân mà còn bảo vệ cho những người xung quanh nên việc nhận diện người đeo khẩu trang sai cách thật sự rất quan trọng
Ứng dụng này cần phải tăng tốc vì mô hình được huấn luyện trên một mạng convolutional neural network
