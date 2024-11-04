class object:
    def __init__(self):
        self.vector = [0,0,0,0]
    
    def change(self, vector):
        vector[0] = 1
        print(self.vector)
    
    def test(self):
        self.change(self.vector)

print('hello')
if __name__ == '__main__':
    t = object()
    t.test()