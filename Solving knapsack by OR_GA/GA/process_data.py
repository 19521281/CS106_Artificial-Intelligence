def str_to_item(s):
    s = [int(el) for el in s.split()]
    return (s[0], s[1])


def get_data(file):
    with open(file, 'r') as reader:
        lines = reader.readlines()
        lines = map(lambda s: s.replace('\n', ''), lines)
        lines = filter(lambda s: s != '', lines)
        lines = list(lines)
        c = int(lines[1])
        items = lines[2:]
        items = list(map(str_to_item, items))
        return items, c


def main():
    items, c = get_data('test.kp')
    print(items[0])


if __name__ == "__main__":
    main()
