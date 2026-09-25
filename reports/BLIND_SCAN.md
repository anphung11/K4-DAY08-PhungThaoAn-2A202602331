# Quét độc lập trước khi xem pre-label

Frame: 0107

Số xe nhìn thấy bằng mắt: Khoảng 25-27

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 
- Vị trí 1 (Dễ bỏ sót hoặc gộp sai): Khu vực cụm đèn xe (đỏ và trắng) nối đuôi nhau ở khoảng cách rất xa phía trước. AI rất dễ bỏ sót các xe này vì kích thước pixel quá nhỏ, hoặc vẽ gộp nhiều xe sát nhau thành một bounding box duy nhất.   
- Vị trí 2 (Dễ vẽ sai kích thước box): Các xe đi ngược chiều ở làn trái đang bật đèn pha chiếu rất chói. AI dễ bị lừa bởi quầng sáng chói (glare/flare) và vẽ box bao trùm cả vùng sáng thay vì bám sát khung thân xe thực tế. Ngoài ra, chiếc xe chỉ lộ một góc nhỏ ở mép dưới cùng của khung hình cũng là vị trí AI dễ đoán sai. 

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
