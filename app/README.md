# TNMT API Service

## API Endpoints
0. `GET /`
1. `POST /classify`
2. `POST /summarize`
3. `POST /sum-cls`


## API Input Format
3 POST requests has the same following format:
```bash
[
    {
        "id"        : "1",
        "title"     : "Trách nhiệm của ... cầm quân đội tuyển Việt Nam",
        "anchor"    : "Lãnh đạo VFF ... U23 Việt Nam",
        "content"   : "Lãnh đạo VFF cho biết, ... đấu trường khu vực đến châu lục."
    },
    {
        "id"        : "2",
        "title"     : "Hồ sơ Luật Địa chất và Khoáng sản cơ bản đáp ứng yêu cầu",
        "anchor"    : "(TN&MT) - Tại cuộc họp ... chiều 10/1",
        "content"   : "Chủ trì cuộc họp ... trình Chính phủ xem xét, quyết định."
    }
]
```

## API Output Format
```bash
[
    {
        "id"        : "1",                          
        "topic"     : "Không",                           
        "sub_topic" : [],                
        "aspect"    : [],         
        "sentiment" : "Không",                      
        "province"  : [],
    },
    {
        "id"        : "2",
        "summary"   : "Ngày 10/1 tại Hà Nội, ... những điểm nổi bật của dự thảo.",
        "topic"     : "Có",
        "sub_topic" : ["Địa chất - Khoáng sản"],
        "aspect"    : ["chính sách, quản lý", "Luật sửa đổi"],
        "sentiment" : "Trung tính",
        "province"  : ["Hà Nội"]
    }
]
```