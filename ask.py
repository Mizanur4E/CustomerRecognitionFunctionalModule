class ask_echo():

    def __init__(self):
        self.name = 'ask_echo'

    def __str__(self):
        return self.name

    def reply(self, question):
        print(f'Replying to the question: {question}')
        return 0