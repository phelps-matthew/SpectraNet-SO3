# To Do
- [X] convert healpix tensorflow operations to pytorch
- [X] fix in-place operation in `predict_probability` preventing gradient backward propagation
- [] Add gradient ascent to `predict_rotation` using argmax(logits) as initialization
- [] Add positional encoding
- [] np arrays to cuda check
- [] numpy vs torch speed test in `generate_healpix_grid`
- [] train/eval mode in so3pdf
- [] `output_pdf` are these really normalized?
- [] output generate queries as torch?
