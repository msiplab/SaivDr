class InvalidDirection(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidTargetChannels(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)   

class InvalidMode(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidMus(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidNumberOfChannels(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidPolyPhaseOrder(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidNumberOfVanishingMoments(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)

class InvalidNumberOfLevels(Exception):
    def __init__(self,msg):
        super().__init__(self,msg)