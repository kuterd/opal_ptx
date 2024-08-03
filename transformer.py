import ast
import inspect

TYPE_TO_REG = {"f64": "b64", "s64": "b64", "u64": "b64", "f32": "b32", "u32": "b32"}


class KernelScope:
    def __init__(self, kernel_builder):
        self.kernel_builder = kernel_builder

    def __enter__(self):
        pass

    def __exit__(self, a, b, c):
        self.kernel_builder.end_block()


class KernelBlock:
    def __init__(self):
        self.instructions = []


class KernelBuilder:
    def __init__(self):
        self.blocks = [KernelBlock()]
        self.block_stack = [self.blocks[0]]
        self.variables = {}
        self.tmp_count = 0

    def If(self, cond_predicate):
        # TODO: process condition and push
        return KernelScope(self)

    def While(self, cond_predicate):
        # TODO: process condition
        return KernelScope(self)

    def new_variable(self, name, typ):
        if name not in self.variables:
            self.variables[name] = typ
            return name
        i = 2
        while True:
            nname = f"{name}_{i}"
            if nname not in self.variables:
                return self.new_variable(nname, typ)
            i += 1

    def tmp_variable(self, typ):
        result = self.new_variable(f"tmp_variable_{self.tmp_count}", typ)
        self.tmp_count += 1
        return result

    def append_instruction(self, instruction):
        self.block_stack[-1].append(instruction)


class OpalExpressionAnalyser:
    """
    Differentiate PTX and python expressions
    """

    def __init__(self):
        self.analysis_cache = {}
        self.current_variables = None  # opal variables

    def _is_opal_expression(self, node):
        if isinstance(node, ast.BinOp):
            return self._Binop(node)
        if isinstance(node, ast.Name):
            return self._Name(node)
        if isinstance(node, ast.Call):
            return self._Call(node)
        if isinstance(node, ast.Compare):
            return self._Compare(node)

        print(type(node))
        return False

    def is_opal_expression(self, node, opal_variables=None):
        if node in self.analysis_cache:
            return self.analysis_cache[node]

        if opal_variables:
            self.current_variables = opal_variables

        cached = self._is_opal_expression(node)
        self.analysis_cache[node] = cached
        # print("OpalExp", node, cached)

        return cached

    def _Name(self, node):
        name = node.id
        print("Checking name", name)
        return name in self.current_variables

    def _Binop(self, node):
        if self.is_opal_expression(node.left):
            return True

        if self.is_opal_expression(node.right):
            return True
        return False

    def _Compare(self, node):
        pass

    def _Call(self, node):
        # ptx ... name () expressions
        def _root(node):
            if isinstance(node, ast.Name):
                return node
            if isinstance(node, ast.Attribute):
                return _root(node.value)
            return node

        root = _root(node.func)
        if not isinstance(root, ast.Name):
            return False
        return root.id == "ptx"


class OpalTransformer(ast.NodeTransformer):
    def __init__(self):
        self.opal_variables = {}
        self.analyser = OpalExpressionAnalyser()

    def _new_tmp_variable(self, typ):
        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="tmp_variable",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=typ)],
            keywords=[],
        )
        return exp

    def _new_variable(self, name, typ):
        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="new_variable",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=name), ast.Constant(value=typ)],
            keywords=[],
        )
        return exp

    def visit_ptxExpression(self, node):
        # Convert a ptx expression and store the result in a variable object.
        pass

    def visit_While(self, node):
        if not self.analyser.is_opal_expression(node.test, self.opal_variables):
            return node

        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="While",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
        return ast.With(items=[ast.withitem(context_expr=exp)], body=node.body)

    def _visit_body(self, body):
        for i, node in enumerate(body):
            body[i] = self.generic_visit(node)

    def visit_If(self, node):
        print(self.opal_variables)
        if not self.analyser.is_opal_expression(node.test, self.opal_variables):
            print("Not opal expression")
            return node
        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="If",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )
        return ast.With(items=[ast.withitem(context_expr=exp)], body=node.body)

    def visit_Assign(self, node):
        # TODO: Convert variable assignments.
        return node

    def visit_AnnAssign(self, node):
        # TODO: Handle buffer assignments?
        if not isinstance(node.target, ast.Name):
            return node

        # TODO: Handle more advanced opal types.
        if not isinstance(node.annotation, ast.Name):
            return node

        name = node.target.id
        typ = node.annotation.id
        self.opal_variables[name] = typ
        print("New variable", name, typ)
        return ast.Assign(targets=[node.target], value=self._new_variable(name, typ))

    def visit_FunctionDef(self, node):
        # Rewrite function arguments to add kernel_builder
        node.args.args = [
            ast.arg(
                arg="kernel_builder", lineno=node.lineno, col_offset=node.col_offset
            )
        ] + node.args.args
        # node.name = "transformed_function"
        node.decorator_list = []
        return self.generic_visit(node)


def kernel():
    """
    The opal kernel decorator.
    """

    def deco(func):
        function = ast.parse(inspect.getsource(func))
        visitor = OpalTransformer()
        function = visitor.visit(function)
        function = ast.fix_missing_locations(function)
        print("Unparsed", ast.unparse(function))
        transformed_code = compile(function, filename="<ast>", mode="exec")
        local_namespace = {}
        exec(transformed_code, func.__globals__, local_namespace)
        return local_namespace[func.__name__]

    return deco
