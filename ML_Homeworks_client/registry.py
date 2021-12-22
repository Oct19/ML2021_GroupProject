# -*- coding: utf-8 -*-

import requests
import time

def main(ip, port, sid, token, reg_info):
    url = "http://%s:%s/jsonrpc" % (ip, port)

    payload = {
        "method": "registry",
        "params": [reg_info],
        "jsonrpc": "2.0",
        "id": 0,
    }
    response = requests.post(
        url,
        json=payload,
        headers={"token": token, "sid": sid}
    ).json()

    if "auth_error" in response:
        print("您的认证信息有误")
        print(response["auth_error"])
    elif "error" not in response:
        print("注册结果：")
        info = response['result']
        print(info)
    else:
        print("提交存在问题，请查看error信息")
        # print(response["error"]["data"]["message"])


if __name__ == "__main__":
    # registry_info需要修改！！！
    # 学号, 姓名, token(密码), email
    registry_info = ['BY2110227', '蔡定坤', 'BY2110227', 'caidingkun@outlook.com']
    # IP 不需要修改
    ip = "115.236.52.125"
    # 端口不需要修改
    port = "4000"
    # 改成你的学号
    sid = "BY2110227"
    # 改成你的口令
    token = "BY2110227"
    for i in range(2):
        print(registry_info)
        a = input('请确认注册信息(%d/2)，你只能注册一次！: (Y/ N)' % (i + 1))

        if a == 'Y':
            continue
        else:
            print('取消注册')
    # time.sleep(3)
    print('注册中……')
    main(ip, port, sid, token, registry_info)
    print('注册完成')
