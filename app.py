# -*- coding:utf-8 _*-
"""
# author : Azhe
# github : https://github.com/LouisAzhe
# time: 2022/7/20 p.m 11:59
# file: app.py
"""
import uvicorn
from fastapi import FastAPI, File, Request
from starlette.responses import Response
import torch
import Captcha_Model

device = torch.device("cuda:0")
use_gpu = torch.cuda.is_available()

if use_gpu == True:
    model = torch.load('densenet121_ep50_fulldata.pkl')
    model = model.cuda(device)
else:
    model = torch.load('densenet121_ep50_fulldata.pkl',map_location='cpu')

app = FastAPI(
    title="SKFH Captcha Recognize API",
    description='金控圖型驗證碼辨識使用API',
)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG'])

@app.get("/",tags=['Test Alive'])
def Hello():
    return {"status":200}

@app.post("/files/",tags=["Domestic Judicial"])
async def UploadImage(file: bytes = File(...)):
    """
         副檔名請使用['png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG']
    """
    try:
        with open('image.jpg', 'wb') as image:
            image.write(file)
            image.close()
        rdl1 = Captcha_Model.return_dataloader('image.jpg')
        result = Captcha_Model.test_model(model, rdl1)
        return {"status": 200, "res": result}
    except:
        return {"status": 999, "res": "Something Error!"}

async def Not404Fount(request: Request, exc):
   response = Response('{"Error": "500"}')
   return response
# 添加404
app.add_exception_handler(404, Not404Fount)

if __name__ == "__main__":
    uvicorn.run(app="app:app", host="0.0.0.0", port=9003, reload=True, debug=True , log_config="uvicorn_config.json")

# 啟用FastAPI並指定後方要連通的port
# uvicorn app:app --reload --port 7788