while True:
    try:
        file = open("DoneAll.txt","r")
        print(file.read())
    except KeyboardInterrupt:
        print("Ctrl-C -> Exit")
    except:
        pass
    