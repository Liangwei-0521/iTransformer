import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sensors.external_task import ExternalTaskSensor
import datetime
from datetime import timedelta
import subprocess
import sys
import os


startdate = datetime.datetime.now().strftime("%Y-%m-%d")



def shanxi_price_day_ahead_predict():
    env_name = "lw_iTransformer "
    python_file_path = "/home/tsikuns/projects/iTransformer_predict/run.py"
    command = f"conda run -n {env_name} python {python_file_path}"
    print(command)
    print("-------------------")
    result = subprocess.run(command, shell=True, capture_output=True)
    # 检查运行命令的结果
    if result.returncode == 0:
        print("命令成功执行")
    else:
        print("命令执行失败")
    # 输出命令的标准输出和标准错误
    print(result.stdout.decode())
    print(result.stderr.decode())
    pass

# 创建中国时区对象
china_tz = datetime.timezone(datetime.timedelta(hours=8), name="China")

# 默认参数
default_args = {
    "owner": "山西",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime.datetime(2024, 7, 15, 15, tzinfo=china_tz),
}

# DAG定义
dag = DAG(
    "基于iTransformer模型的山西日前价格预测",
    default_args=default_args,
    description="根据山西三天(包含当天)边界信息滚动预测明日出清-日前出清价格",
    schedule_interval='00 09 * * *',  # timedelta(days=1),
    catchup=False,
)

start = DummyOperator(task_id="start", dag=dag)

task1 = PythonOperator(
    task_id="shanxi_price_day_ahead_predict",
    python_callable=shanxi_price_day_ahead_predict,
    dag=dag,
)


# 任务依赖
start >> task1 