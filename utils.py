def get_key(value, dictionary):
    ''' get key from value in dictionary '''
    for k,v in dictionary.items():
        if value == v:
            return k