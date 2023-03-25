import torch
import yaml

from abc import ABC
from typing import List


class Task(ABC):
    def solve(self):
        """
        Function to implement your solution, write here
        """

    def evaluate(self):
        """
        Function to evaluate your solution
        """


class Task1(Task):
    """
        Calculate, using PyTorch, the sum of the elements of the range from 0 to 10000.
    """

    def __init__(self) -> None:
        self.task_name = "task1"

    def solve(self):
        t = torch.arange(0, 10000)
        return t.sum()

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.item()}}


class Task2(Task):
    """
        Solve optimization problem: find the minimum of the function f(x) = ||Ax^2 + Bx + C||^2, where
        - x is vector of size 8
        - A is identity matrix of size 8x8
        - B is matrix of size 8x8, where each element is 0
        - C is vector of size 8, where each element is -1
        Use PyTorch and autograd function. Relative error will be less than 1e-3

        Solution here is x, converted to the list(see submission.yaml).
    """

    def __init__(self) -> None:
        self.task_name = "task2"

    def solve(self):
        lr = 1e-3

        x = torch.rand(8, requires_grad=True)
        A = torch.eye(8)
        B = torch.zeros(8, 8)
        C = torch.Tensor(8).fill_(-1)

        for idx in range(5000):
            x.grad = None
            L = (A @ (x ** 2) + B @ x + C).norm() ** 2
            L.backward()
            with torch.no_grad():
                x -= x.grad * lr
        return x

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class Task3(Task):
    """
        Solve optimization problem: find the optimal parameters of the linear regression model, using PyTorch.
        train_X = [[0, 0], [1, 0], [0, 1], [1, 1]],
        train_y = [1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746]
        text_X = [[0, -1], [-1, 0]]
        User PyTorch. Relative error will be less than 1e-1

        Solution here is test_y, calculated from test_X, converted to the list(see submission.yaml).
    """

    def __init__(self) -> None:
        self.task_name = "task3"

    def solve(self):
        train_X = torch.Tensor([[0, 0], [1, 0], [0, 1], [1, 1]])
        train_y = torch.Tensor([1.0412461757659912, 0.5224423408508301, 0.5145719051361084, 0.052878238260746])
        test_X = torch.Tensor([[0, -1], [-1, 0]])

        class Linear(torch.nn.Module):
            def __init__(self, ):
                super().__init__()

                self.layer = torch.nn.Linear(2, 1)

            def forward(self, x):
                return self.layer(x)

        model = Linear()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), 1e-3)

        for idx in range(10000):
            optimizer.zero_grad()
            pred_y = model(train_X)
            loss = criterion(pred_y[:, 0], train_y)
            loss.backward()
            optimizer.step()

        return model(test_X)

    def evaluate(self):
        solution = self.solve()

        return {self.task_name: {"answer": solution.tolist()}}


class HW(object):
    def __init__(self, list_of_tasks: List[Task]):
        self.tasks = list_of_tasks
        self.hw_name = "submission"

    def evaluate(self):
        aggregated_tasks = []

        for task in self.tasks:
            aggregated_tasks.append(task.evaluate())

        aggregated_tasks = {"tasks": aggregated_tasks}

        yaml_result = yaml.dump(aggregated_tasks)

        print(yaml_result)

        with open(f"{self.hw_name}.yaml", "w") as f:
            f.write(yaml_result)


hw = HW([Task1(), Task2(), Task3()])
hw.evaluate()