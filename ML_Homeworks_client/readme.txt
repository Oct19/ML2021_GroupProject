文件结构：
    |-- registry.py
    |-- client.py
    |-- leaderboard.py

leaderboard以及提交规则：
leaderboard取最后一次提交作为排行榜的排名
提交次数不限制，但是提交时间间隔需要大于60s

registry.py
    使用方法：python registry.py

    该程序进行注册竞赛账号
    注意！！：对于每个学号，你将只能注册一次，这个学号将会参与到竞赛排名中，
            你的作业中应当包含这个学号的竞赛排名截图。

    你需要修改的包括：
        registry_info = [真实学号, 姓名, token（密码）, 邮箱]


client.py：
    注意：在使用client之前，你需要使用registry.py进行注册！！
    使用方法：python client.py

    该程序用于提交文件到服务器上，接收到服务器返回的结果，并显示结果。
    提交的结果会记录在服务器上，并参与排名。

    你需要修改的包括：
        problem = 你参与的任务（作业题）
        sid =     你的学号
        token =   你的token，也是密码
    对于
        "Action_evaluate",
        'FoodPredict_evaluate',
        'StoreSale_evaluate',
        'Toxicity_evaluate',
        'CarDemand_evaluate',
        'FineGrainedCar_evaluate',
        'Traffic_evaluate'   这些任务，
    你需要将submission.txt放在client.py所处的当前目录。
    对于"Mask_evaluate"，
    你需要将文件夹submission放在当前目录。

leaderboard.py
    使用方法：python leaderboard.py

    该程序用来查看任务排名

    你需要修改的包括：
        problem = 你参与的任务（作业题）
        sid =     你的学号
        token =   你的token，也是密码