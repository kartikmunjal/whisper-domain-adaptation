from scripts.update_extension_docs import block,ci
def test_block_preserves_surroundings():
 assert block('a<S>old<E>b','<S>','<E>','new')=='a<S>\nnew\n<E>b'
def test_ci_formats_scaled_interval():
 assert ci([.1,.05,.2],100,' points')=='10.00 points (95% CI 5.00–20.00)'
