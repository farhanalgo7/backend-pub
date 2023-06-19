import random
import json

from locust import HttpUser, TaskSet, task, between

search_strings_list = ["TCS", "TCS hiring", "HDFC bank loans", "SBI recruitment", "Bank of India",
                        "State Bank of India", "Bank of New York"]

qna_strings_list = ["What is the interest rate of SBI bank?", "Will TCS hire this year?",
                    "How much did HDFC earn this financial year?", "Is Infosys hiring freshers?",
                    "What is the profit of TCS this year?"]

data = dict()

class User(HttpUser):
    wait_time = between(2, 5)

    @task(2)
    def test_semantic_search(self):
        headers = {'Content-type': 'application/json'}
        data["search_string"] = random.choice(search_strings_list)

        self.client.get("/search",
                    json=data,
                    headers=headers, 
                    name="Simple Search")

    @task()
    def test_question_answer_extraction(self):
        headers = {'Content-type': 'application/json'}
        data["search_string"] = random.choice(qna_strings_list)

        self.client.get("/search",
                    json=data,
                    headers=headers, 
                    name="Question Answer")