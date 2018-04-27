import re
def generate_label_views(kzip_path):
    # define pathes to skeleton(kzip file) and mesh(only vertices, npy file)
    # TODO: retrieve vert_fname and ind_fname from kzip_path, use package 're" (regular expression) for that
    # retrieve the identification number of  sso
    #kzip_path2int = int(kzip_path)
    retr_names = int(re.findall("/(\d+).", kzip_path)[0])
    #looking for 8 digits in a row in string kzip_path(??)
    #is there moore eleborate way to code 8 digits in the rows indstead of typing d letters ?
    #re.search vs re.match??

    # print(retr_names)
    # retr_names_int2str = str(retr_names)
    vert_fname = label_file_folder + str(retr_names) + "_vert.npy"
    ind_fname = label_file_folder + str(retr_names) + "_ind.npy"
    print(vert_fname)
    print(ind_fname)

if __name__ == "__main__":
    #print(generate_label_views)
    label_file_folder = "/u/shum/axon_label_files/"
    kzip_path = label_file_folder + "/28985344.001.k.zip"
    label_views = generate_label_views(kzip_path)
