class RawValue:

    def __init__(self, value, count = None):

        self.isSingleNumber=False
        self.isArrayNumber = False
        self.isSimpleImageNumber = False
        self.isArrayImageNumber = False
        self.isSimpleImageChar = False
        self.isArrayImageChar = False
        self.isSimpleChar = False
        self.isArrayChar = False
        self.isCategoric = False
        self.value=value
        self.isPosition = False
        self.name = None
        self.count = None

        def __str__(self):
            return str(self.value)

class Position(RawValue):
    def __init__(self, value, count):
        super().__init__(value, count)
        self.isPosition = True
        self.name = 'position'

class SingleNumber(RawValue):
    def __init__(self, value):
        super().__init__(value, 1)
        self.isSingleNumber = True
        self.name = 'single_number'


class ArrayNumber(RawValue):
    def __init__(self, value, count):
        super().__init__(value, count)
        self.isArrayNumber = True
        self.name = 'array_number'


class SingleImageNumber(RawValue):
    def __init__(self, value):
        super().__init__(value, 1)
        self.isSingleImageNumber = True
        self.name = 'single_image_number'


class ArrayImageNumber(RawValue):
    def __init__(self, value, count):
        super().__init__(value, count)
        self.isArrayImageNumber = True
        self.name = 'array_image_number'
    def __str__(self):
        return str(self.value)

class SingleImageChar(RawValue):
    def __init__(self, value):
        super().__init__(value, 1)
        self.isSingleImageChar = True
        self.name = 'single_image_char'
    def __str__(self):
        return str(self.value)

class ArrayImageChar(RawValue):
    def __init__(self, value, count):
        super().__init__(value, count)
        self.isArrayImageChar = True
        self.name = 'array_image_char'
    def __str__(self):
        return str(self.value)

class SingleChar(RawValue):
    def __init__(self, value):
        super().__init__(value, 1)
        self.isSingleChar = True
        self.name = 'single_char'
    def __str__(self):
        return str(self.value)

class ArrayChar(RawValue):
    def __init__(self, value, count):
        super().__init__(value, count)
        self.isArrayChar = True
        self.name = 'array_char'
