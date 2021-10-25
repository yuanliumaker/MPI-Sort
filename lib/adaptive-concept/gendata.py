import numpy as np

np.random.randint(2, size=10).astype(np.uint32).tofile('ciccio.raw')
np.arange(10, dtype = np.uint32).tofile('ciccia.raw')
(313 * np.ones(7, dtype = np.uint32)).tofile('cicche.raw')
