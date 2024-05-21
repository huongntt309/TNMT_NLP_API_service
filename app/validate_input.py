
def check_duplicate_ids(array_data):
    # Sử dụng set để theo dõi các id đã gặp và một set khác để theo dõi các id trùng lặp
    seen_ids = set()
    duplicate_ids = set()

    # Vòng lặp để xác định các id trùng lặp
    for item in array_data:
        item_id = item.get("id", "")
        if item_id in seen_ids:
            duplicate_ids.add(item_id)
        else:
            seen_ids.add(item_id)

    # Nếu có id trùng lặp, return response luôn
    if duplicate_ids:
        return {
            "status": "error",
            "message": "Duplicate ids found",
            "duplicate_ids": list(duplicate_ids)
        }
    
    return {
        "status": "success",
        "message": "No duplicate ids found"
    }