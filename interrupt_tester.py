from pynput import keyboard
import random

class MyException(Exception): pass

class TypeListener:
    def __init__(self):
        self.count = 0
        self.limit = random.randint(5, 15)

    def on_press(self, key):
        if key != keyboard.Key.enter:
            self.count += 1
            if self.count >= self.limit:
                print('Stop')
                raise MyException(key)

def start_program():
    listener = TypeListener()
    with keyboard.Listener(on_press=listener.on_press) as listener:
        try:
            listener.join()
        except MyException as e:
            pass

start_program()
