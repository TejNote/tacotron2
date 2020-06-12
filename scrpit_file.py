
writeFile = open("kss_train_filelist_v3.txt", 'w')
readFile = open("transcript.v.1.4.txt", 'r')
while True:
    line = readFile.readline()
    if not line: break
    result = line.split("|")
    path = "./kss/resample/" + result[0]
    script = result[1]
    print(path, script)
    writeFile.write(path + "|" + script + "\n")

readFile.close()
writeFile.close()