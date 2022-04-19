import tushare
import pandas

def main():
    tocken = open('./tocken.txt').read()
    tushare.set_token(tocken)

if __name__ == '__main__':
    main()