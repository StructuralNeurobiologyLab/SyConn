import collections

def dict_functionality():
     test_dict = collections.defaultdict(list)
     tab = [2, 5, 8]

     for i in range(10):
         test_dict[i] = [i, 2*i, 3*i]

     dict2 = test_dict[1, 3, 4]
     print("test_dict= ", test_dict)
     print("dict2= ", dict2)



def main():
    dict_functionality()


if __name__ == "__main__":
    main()