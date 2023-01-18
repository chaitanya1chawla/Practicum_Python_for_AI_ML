import pandas as pd


def convert_to_dt(*args, dtype, debug=False) -> list:
    cast_list = []
    try:
        for i in range(len(args)):
            cur_val = args[i]
            new_val = dtype(cur_val)
            cast_list.append(new_val)
    except ValueError as exc:
        if debug:
            msg = f"ErrorMessage : Can't convert '{cur_val}' at index {i}"
            print(msg)

        raise Exception(exc)

    print(cast_list)
    return cast_list


def replace_all(string: str) -> str:
    string = string.replace("’", "")
    string = string.replace("'", "")
    # string = string.replace("(", "")
    # string = string.replace(")", "")
    return string


if __name__ == '__main__':
    print("Executing as main program")
    inp = input()  # [1 , 4 , 5]
                # [ ’5 ’, ’a ’ , ’15.1516 ’]
    dtype = int
    debug = True
    print("encountered value error (main program)")
    try:
        inp = replace_all(inp)
        inp = inp.split(',')
        args = []
        for ip in inp:
            ip = ip.replace('[', '')
            ip = ip.replace(']', '')
            ip = ip.strip()
            args.append(ip)

        cast_list = convert_to_dt(*args, dtype=dtype, debug=debug)
    except:
        print("encountered value error (main program)")

    #
    #
    # cast_list = convert_to_dt((1, 0), "a", 15.1516, dtype=str)
    #
    # # cast_list = convert_to_dt(5, "a", dtype=int, debug=False)
    # cast_list = convert_to_dt(5, "a", dtype=int, debug=True)

# >>> convert_to_dt ("1", 4, 5.0 , dtype =int)
# 2 [1, 4, 5]
# 3 >>> convert_to_dt ((1 ,0) , "a", 15.1516 , dtype =str)
# 4 [’(1, 0) ’, ’a’, ’15.1516 ’]
# 5 >>> convert_to_dt (5, "a", dtype =int , debug = False )
# 3.7
