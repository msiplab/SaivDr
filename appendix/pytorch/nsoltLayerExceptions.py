class InvalidDirection(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidTargetChannels(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)        
