import time
import enlighten

m = enlighten.get_manager()

pbar = m.counter(
    total=100,
    units='',
    descr='Description text',
    leave=True,
    color='purple'
)

for i in range(100):
    time.sleep(0.1)
    pbar.update()

pbar.close()
