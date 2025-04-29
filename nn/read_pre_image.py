import pickle

if __name__ == "__main__":
    infile = open("../nn/pursuit_evasion_coarse_pre_image_h", 'rb')
    pre_image = pickle.load(infile)
    for j in range(0, len(pre_image)):
        print("-- class " + str(j))
        if len(pre_image[j]) != 0:
            for constraints in pre_image[j]:
                # constraints expressed B+Ax > 0
                print(constraints)
