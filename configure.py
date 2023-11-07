STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

#---

patch_size   = 16
num_pixel    = (patch_size)**2
pixel_scale  = 2.0  #1.0  #0.62=36/58 #1.0

max_patch_row_col = 500
max_num_patch = 600


#---

vocab_size = 193
max_length = 300 #278 #275


#---

patch_dim  = 768
text_dim   = 384
num_layer = 3
num_head  = 8
ff_dim = 1024