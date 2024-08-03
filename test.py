import transformer


@transformer.kernel()
def test():
    test_var: u64 = 123
    print(f"Variable is %{test_var}")
    if test_var < 123:
        pass


# test(transformer2.KernelBuilder()
