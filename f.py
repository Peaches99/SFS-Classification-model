#write me a lambda function that checks if a string is hashed and then returns a bool


    
    def is_hashed(string):
        return string.startswith('$2b$')
    
    #write me a lambda function that checks if a string is hashed and then returns a bool
    is_hashed = lambda string: string.startswith('$2b$')
    
    is_hashed = {if(string.startswith('$2b$')): return True else: return False}