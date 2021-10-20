import threading
import os
import urllib.request


class Downloader(threading.Thread):
    def __init__(self, id, url, path):
        threading.Thread.__init__(self)
        self.id = id
        self.url = url
        self.path = path

    def run(self):
        print("Thread {} here".format(self.id))
        urllib.request.urlretrieve(self.url, self.path)
        print("Exiting thread {}".format(self.id))


if __name__ == "__main__":
    urls = ["https://pliki.ptwp.pl/pliki/03/93/05/039305_r1_300.jpg",
            "https://mantas.info/wp/wp-content/uploads/simple_esn/MackeyGlass_t17.txt",
            "https://pbs.twimg.com/media/DY7ATccWkAEzTAU.jpg",
            "https://d2lljesbicak00.cloudfront.net/merida-v2/crud-content-img//db-global/2021/product-pages-tags/21-merida-road-bikes-reacto-my2021-parallax.jpg"]
    directory = "downloads"

    threads = []
    for i, url in enumerate(urls):
        threads.append(Downloader(i, url, os.path.join(
            directory, os.path.basename(url))))
        threads[-1].start()

    for t in threads:
        t.join()
