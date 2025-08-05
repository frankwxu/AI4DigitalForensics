class Student:
    school = "High School"

    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

    def study(self):
        return f"{self.name} is studying!"