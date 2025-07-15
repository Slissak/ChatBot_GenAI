from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
# from diagrams.onprem.analytics import Elasticsearch
from diagrams.custom import Custom

with Diagram("Job Interview Bot - Multi-Agent System", show=True, direction="TB"):

    user = Custom("User", "./icons/user.png")

    with Cluster("Supervisor Agent"):
        supervisor = Custom("Supervisor", "./icons/supervisor_agent.png")

        with Cluster("Sub-agents"):
            info_agent = Custom("Info Agent\n(RAG)", "./icons/info_agent.png")
            schedule_agent =  Custom("Schedule Agent\n(SQL DB)", "./icons/DB_agent.png")
            exit_agent = Custom("Exit Agent\n(Fine-tuned)", "./icons/exit_agent.png")

        supervisor >> Edge(dir="both") >> info_agent
        # info_agent >> supervisor
        supervisor >> Edge(dir="both") >> schedule_agent
        # schedule_agent >> supervisor
        supervisor >> Edge(dir="both") >> exit_agent
        # exit_agent >> supervisor

    user >> Edge(dir="both") >> supervisor
    # supervisor >> user
