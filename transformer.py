import ast
import inspect
import struct
from enum import Enum


class BasicTypes(str, Enum):
    u64 = "u64"
    s64 = "s64"

    u32 = "u32"
    s32 = "s32"

    f64 = "f64"
    f32 = "f32"
    f16 = "f16"
    f8 = "f8"


float_types = ["f64", "f32", "f16", "f8"]

TYPE_TO_REG = {
    "b32": "b32",
    "b64": "b64",
    "s64": "b64",
    "u64": "b64",
    "u32": "b32",
    "s32": "b32",
    "f64": "f64",
    "f32": "f32",
    "f16": "f16",
    "f8": "f8",
    "pred": "pred",
}


def double_to_hex(d):
    # Pack the double into a byte array (IEEE 754 double-precision format)
    byte_array = struct.pack(">d", d)

    # Convert the byte array to an integer
    int_value = int.from_bytes(byte_array, byteorder="big")

    # Convert the integer to a hexadecimal string
    hex_string = format(int_value, "016x")

    return hex_string


class OpalType:
    def get_reg_type(self):
        raise NotImplementedError()

    def get_fundamental_type(self):
        return self.get_reg_type()


class BasicType(OpalType):
    def __init__(self, type_name):
        self.type_name = type_name

    def get_fundamental_type(self):
        return self.type_name

    def get_reg_type(self):
        return TYPE_TO_REG[self.type_name]


class RegisterVector(OpalType):
    def __init__(self, type_name, length):
        self.length = length
        self.type_name = type_name

    def get_fundamental_type(self):
        return self.type_name

    def get_reg_type(self):
        return TYPE_TO_REG[self.type_name]


class SharedMemoryType(OpalType):
    def __init__(self, type_name, length):
        self.type_name = type_name
        self.length = length

    def get_fundamental_type(self):
        return "u32"

    def get_reg_type(self):
        return TYPE_TO_REG["u32"]  # Pointer type


class KernelBlock:
    def __init__(self, name):
        self.instructions = []
        self.name = name

    def generate(self):
        result = f"${self.name}:\n"
        for inst in self.instructions:
            result += f"\t{inst}\n"
        return result

    def dump(self):
        print(self.generate())


class KernelScope:
    def __init__(self, kernel_builder, block_name):
        self.kernel_builder = kernel_builder
        self.block_name = block_name

    def __enter__(self):
        self.kernel_builder.blocks.append(KernelBlock(self.block_name))
        self.kernel_builder.block_stack.append(self.kernel_builder.blocks[-1])

    def __exit__(self, a, b, c):
        self.kernel_builder.block_stack.pop()

    def __repr__(self):
        return self.block_name


class KernelBuilder:
    def __init__(self):
        self.block_name_counter = 1
        self.blocks = [KernelBlock("entry")]
        self.block_stack = [self.blocks[0]]

        self.variables = {}
        self.shared_memory_entries = []
        self.tmp_count = 0
        self.name = None
        self.args = None

    def generate_header(self, name, args):
        self.name = name
        self.args = args

    def Block(self, prefix=""):
        result = KernelScope(self, f"{prefix}_block_{self.block_name_counter}")
        self.block_name_counter += 1
        return result

    def _allocate_name(self, name):
        if name not in self.variables:
            return name

        # NOTE: This is not super efficient.
        i = 2
        while True:
            nname = f"{name}_{i}"
            if nname not in self.variables:
                return nname
            i += 1

    def new_variable(self, name, typ):
        name = self._allocate_name(name)
        self.variables[name] = typ
        return f"%{name}"

    def allocate_shared(self, name, typ, length):
        name = self._allocate_name(name)
        self.variables[name] = "shared"
        self.shared_memory_entries.append((name, typ, length))

        return f"{name}"

    def new_register_vector(self, name, base_typ, length):
        result = []
        for i in range(length):
            # This is super ugly use the register vector thing in ptx.
            nname = f"{name}_{i}"
            nname = self._allocate_name(nname)
            result.append("%" + nname)
            self.variables[nname] = base_typ

        return result

    def tmp_variable(self, typ):
        result = self.new_variable(f"tmp_variable_{self.tmp_count}", typ)
        self.tmp_count += 1
        return result

    def append_instruction(self, inst):
        self.block_stack[-1].instructions.append(inst + ";")

    def generate(self, target, version="8.5"):
        result = f"""
        .version {version}
        .target {target}
        .address_size 64
        """

        args = ",".join([f".param .{typ} {name}" for name, typ in self.args])
        result += f".visible .entry {self.name}({args}) " + "{\n"

        for name, typ in self.variables.items():
            if typ != "shared":
                result += f"\t.reg .{TYPE_TO_REG[typ]} %{name};\n"
        for name, typ, length in self.shared_memory_entries:
            result += f"\t .shared .{typ} {name}[{length}];\n"

        for i, block in enumerate(self.blocks):
            result += block.generate()
        result += "}"
        return result

    def dump(self):
        print(self.generate())


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
        elif isinstance(node, ast.BoolOp):
            return self._BoolOp(node)
        elif isinstance(node, ast.UnaryOp):
            return self._Unary(node)
        elif isinstance(node, ast.Name):
            return self._Name(node)
        elif isinstance(node, ast.Call):
            return self._Call(node)
        elif isinstance(node, ast.Compare):
            return self._Compare(node)
        elif isinstance(node, ast.Subscript):
            return self._Subscript(node)

        return False

    def is_opal_expression(self, node, opal_variables=None):
        if node in self.analysis_cache:
            return self.analysis_cache[node]

        if opal_variables:
            self.current_variables = opal_variables

        cached = self._is_opal_expression(node)
        self.analysis_cache[node] = cached

        return cached

    def _Name(self, node):
        name = node.id
        return name in self.current_variables

    def _BoolOp(self, node):
        for val in node.values:
            if not self.is_opal_expression(val):
                return False
        return True

    def _Binop(self, node):
        if self.is_opal_expression(node.left):
            return True

        if self.is_opal_expression(node.right):
            return True
        return False

    def _Unary(self, node):
        return self.is_opal_expression(node.operand)

    def _Compare(self, node):
        if self.is_opal_expression(node.left):
            return True
        for comp in node.comparators:
            if self.is_opal_expression(comp):
                return True
        return False

    def _Subscript(self, node):
        # Register vector addressing.
        return self.is_opal_expression(node.value)

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
    def __init__(self, is_fragment=False):
        self.opal_variables = {}
        self.block_counter = 1
        self.analyser = OpalExpressionAnalyser()
        self.body_stack = []
        self.tmp_python_variables = 0
        self.is_fragment = is_fragment

    def push_body_continuation(self, node, attr):
        self.push_expr(node)
        self.body_stack[-1] = getattr(node, attr)

    def push_expr(self, node):
        self.body_stack[-1].append(node)

    def _new_block(self, prefix="") -> str:
        name = f"block_{self.block_counter}"
        self.block_counter += 1

        exp = ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                    attr="Block",
                    ctx=ast.Load(),
                ),
                args=[ast.Constant(value=prefix)],
                keywords=[],
            ),
        )
        self.push_expr(exp)
        return name

    def _new_tmp_variable(self, typ: OpalType):
        assert isinstance(typ, BasicType)
        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="tmp_variable",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=typ.type_name)],
            keywords=[],
        )
        return exp

    def _new_tmp_variable_statement(self, typ: OpalType) -> str:
        # Create temporary ptx variable and assign it to a temporary python variable.
        name = f"_tmp_{self.tmp_python_variables}"
        exp = ast.Assign(
            targets=[ast.Name(id=name, ctx=ast.Store())],
            value=self._new_tmp_variable(typ),
        )
        self.tmp_python_variables += 1
        self.push_expr(exp)

        return name

    def _new_register_vector(self, name: str, typ_name: str, length: int):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="new_register_vector",
                ctx=ast.Load(),
            ),
            args=[
                ast.Constant(value=name),
                ast.Constant(value=typ_name),
                ast.Constant(value=length),
            ],
            keywords=[],
        )

    def _new_shared_memory_allocation(self, name, typ: SharedMemoryType):
        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="allocate_shared",
                ctx=ast.Load(),
            ),
            args=[
                ast.Constant(value=name),
                ast.Constant(value=typ.type_name),
                ast.Constant(value=typ.length),
            ],
            keywords=[],
        )

    def _new_variable(self, name, typ):
        if isinstance(typ, RegisterVector):
            return self._new_register_vector(name, typ.type_name, typ.length)
        elif isinstance(typ, SharedMemoryType):
            return self._new_shared_memory_allocation(name, typ)

        assert isinstance(typ, BasicType)

        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="new_variable",
                ctx=ast.Load(),
            ),
            args=[ast.Constant(value=name), ast.Constant(value=typ.type_name)],
            keywords=[],
        )
        return exp

    def _insert_ptx_instruction(self, parts):
        part_values = []

        for part in parts:
            if isinstance(part, str):
                part_values.append(ast.Constant(value=part))
            else:
                part_values.append(ast.FormattedValue(value=part, conversion=-1))

        instruction_str = ast.JoinedStr(values=part_values)

        exp = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                attr="append_instruction",
                ctx=ast.Load(),
            ),
            args=[instruction_str],
            keywords=[],
        )
        self.push_expr(ast.Expr(value=exp))

    def ptx_cast(self, from_type: OpalType, to_type: OpalType, arg) -> (str, BasicType):
        to_reg_type = to_type.get_reg_type()
        if isinstance(from_type, SharedMemoryType) and to_reg_type != "b32":
            arg, from_type = self.ptx_cast(from_type, BasicType("u32"), arg)
        from_reg_type = from_type.get_reg_type()

        if to_reg_type == "b64" and from_reg_type == "b32":
            result_name = self._new_tmp_variable_statement(to_type)
            self._insert_ptx_instruction(
                [
                    f"mov.{to_reg_type} ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ", {",
                    ast.Name(id=arg, ctx=ast.Load()),
                    ", 0}",
                ]
            )
            return result_name, to_type
        elif to_reg_type == "b32" and from_reg_type == "b32":
            result_name = self._new_tmp_variable_statement(to_type)
            self._insert_ptx_instruction(
                [
                    f"mov.{to_type.get_fundamental_type()} ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ",",
                    ast.Name(id=arg, ctx=ast.Load()),
                ]
            )
            return result_name, to_type
        elif to_reg_type == "b32" and from_reg_type == "b64":
            result_name = self._new_tmp_variable_statement(to_type)
            self._insert_ptx_instruction(
                [
                    f"cvt.{to_type.get_fundamental_type()}.{from_type.get_fundamental_type()} ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ",",
                    ast.Name(id=arg, ctx=ast.Load()),
                ]
            )
            return result_name, to_type

        elif to_reg_type == "pred":
            result_name = self._new_tmp_variable_statement(to_type)
            self._insert_ptx_instruction(
                [
                    f"setp.{from_type.get_fundamental_type()}.eq ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ",",
                    ast.Name(id=arg, ctx=ast.Load()),
                    ", 1",
                ]
            )
            return result_name, to_type
        elif (
            to_type.get_fundamental_type() in float_types
            and from_type.get_fundamental_type() not in float_types
        ):
            # non-float to float conversion.
            # FIXME: Test

            rounding = ".rn"

            # Float to float conversion
            result_name = self._new_tmp_variable_statement(to_type)
            self._insert_ptx_instruction(
                [
                    f"cvt.{to_type.get_fundamental_type()}.{from_type.get_fundamental_type()}{rounding} ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ", ",
                    ast.Name(id=arg, ctx=ast.Load()),
                ]
            )

            return result_name, to_type

        elif (
            to_type.get_fundamental_type() in float_types
            and from_type.get_fundamental_type() in float_types
        ):
            # Float to float conversion.

            rounding = ""
            if float_types.index(to_type.get_fundamental_type()) > float_types.index(
                from_type.get_fundamental_type()
            ):
                rounding = ".rn"

            # Float to float conversion
            result_name = self._new_tmp_variable_statement(to_type)
            self._insert_ptx_instruction(
                [
                    f"cvt.{to_type.get_fundamental_type()}.{from_type.get_fundamental_type()}{rounding} ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ", ",
                    ast.Name(id=arg, ctx=ast.Load()),
                ]
            )

            return result_name, to_type

        else:
            return arg, to_type

    def _ptx_binop(self, op: str, op_type: OpalType, target: str, left, right):
        # Decide type casts based on the type of left and right.
        op_to_inst = {
            "Add": "add",
            "Sub": "sub",
            "Mult": "mul",
            "Div": "div",
            "RShift": "shr",
            "LShift": "shl",
            "BitAnd": "and",
            "BitXor": "xor",
            "BitOr": "or",
            # TODO: Add Mod
        }

        inst = op_to_inst[op]

        target_type = op_type.get_fundamental_type()
        if inst == "mul" and target_type in ["u32", "s32", "u64", "s64"]:
            inst += ".lo"
        if inst == "div" and target_type in float_types:
            inst += ".rn"
        if inst == "shr" or inst == "shl":
            right = self.ptx_cast(right[1], BasicType("u32"), right[0])
        if inst in ["shl", "xor", "and", "or"]:
            target_type = TYPE_TO_REG[target_type]

        self._insert_ptx_instruction(
            [
                f"{inst}.{target_type} ",
                ast.Name(id=target, ctx=ast.Load()),
                ",",
                ast.Name(id=left[0], ctx=ast.Load()),
                ",",
                ast.Name(id=right[0], ctx=ast.Load()),
            ]
        )

        return target, op_type

    def _binary_cast(
        self, left: (str, OpalType), right: (str, OpalType)
    ) -> ((str, OpalType), (str, OpalType)):
        type_precedence = {
            "f64": 8,
            "f32": 7,
            "f16": 6,
            "s64": 4,
            "u64": 3,
            "s32": 2,
            "u32": 1,
            "pred": 0,
        }
        op_type = left[1]

        if left[1] != right[1]:
            # Implicit type casting is needed!
            if (
                type_precedence[left[1].get_fundamental_type()]
                > type_precedence[right[1].get_fundamental_type()]
            ):
                op_type = left[1]
                right = self.ptx_cast(right[1], op_type, right[0])
            else:
                op_type = right[1]
                left = self.ptx_cast(left[1], op_type, left[0])

        return left, right

    def visit_ptxBinOp(self, node) -> (str, OpalType):
        left = self.visit_ptxExpression(node.left)
        right = self.visit_ptxExpression(node.right)

        left, right = self._binary_cast(left, right)

        op = type(node.op).__name__
        result_name = self._new_tmp_variable_statement(left[1])
        return self._ptx_binop(op, left[1], result_name, left, right)

    def visit_ptxBoolOp(self, node: ast.BoolOp) -> (str, OpalType):
        # TODO: What should be the type of the result?
        def _handle_binary(op, result: str, right: (str, OpalType)):
            pass

        op = type(node.op).__name__
        left = self.visit_ptxExpression(node.values[0])
        result = self._new_tmp_variable_statement(left[1])
        # TODO: move left to the result.

        for val in node.values[1:]:
            ptx_val = self.visit_ptxExpression(val)
            _handle_binary(node.op, result, ptx_val)

        # return

    def visit_ptxUnaryOp(self, node: ast.UnaryOp) -> (str, OpalType):
        op = type(node.op).__name__

        s = self.visit_ptxExpression(node.operand)
        if op == "UAdd":
            return s
        elif op == "USub":
            result_name = self._new_tmp_variable_statement(s[1])
            self._insert_ptx_instruction(
                [
                    f"neg.{s[1].get_fundamental_type()} ",
                    ast.Name(id=result_name, ctx=ast.Load()),
                    ",",
                    ast.Name(id=s[0], ctx=ast.Load()),
                ]
            )
            return result_name, s[1]

        raise NotImplementedError
        # TODO: Implement other ops.

    def visit_ptxConstant(self, node: ast.Constant) -> (str, OpalType):
        val = node.value
        typ = None

        if isinstance(val, float):
            val = "0d" + double_to_hex(val)
            typ = BasicType("f64")
        elif isinstance(val, int):
            typ = BasicType("s64")

        assert typ is not None

        result_name = self._new_tmp_variable_statement(typ)
        self._insert_ptx_instruction(
            [
                f"mov.{typ.type_name} ",
                ast.Name(id=result_name, ctx=ast.Load()),
                f", {str(val)}",
            ]
        )
        return result_name, typ

    def visit_ptxCompare(self, node: ast.Compare) -> (str, OpalType):
        assert len(node.comparators) == 1
        assert len(node.ops) == 1

        left = self.visit_ptxExpression(node.left)
        right = self.visit_ptxExpression(node.comparators[0])

        left, right = self._binary_cast(left, right)
        result_name = self._new_tmp_variable_statement(BasicType("b32"))

        op = type(node.ops[0]).__name__
        op_map = {
            "Gt": "gt",
            "GtE": "ge",
            "Lt": "lt",
            "LtE": "le",
            "Eq": "eq",
            "NotEq": "ne",
        }
        self._insert_ptx_instruction(
            [
                f"set.{op_map[op]}.s32.{left[1].get_fundamental_type()} ",
                ast.Name(id=result_name, ctx=ast.Load()),
                ", ",
                ast.Name(id=left[0], ctx=ast.Load()),
                ", ",
                ast.Name(id=right[0], ctx=ast.Load()),
            ]
        )
        return result_name, BasicType("u32")

    def visit_ptxVariableReference(self, node: ast.Call):
        assert len(node.args) == 1
        assert isinstance(node.func, ast.Name)
        typ = BasicType(node.func.id)

        arg = node.args[0]

        result_name = self._new_tmp_variable_statement(typ)
        self._insert_ptx_instruction(
            [
                f"mov.{node.func.id} ",
                ast.Name(id=result_name, ctx=ast.Load()),
                ", ",
                arg,
            ]
        )

        return result_name, typ

    def visit_ptxSubscript(self, node) -> (str, OpalType):
        assert isinstance(node.value, ast.Name)
        assert node.value.id in self.opal_variables

        typ = self.opal_variables[node.value.id]
        assert isinstance(typ, RegisterVector)

        result_typ = BasicType(typ.type_name)
        result_name = self._new_tmp_variable_statement(result_typ)

        self.push_expr(
            ast.Assign(targets=[ast.Name(id=result_name, ctx=ast.Store())], value=node)
        )

        return result_name, result_typ

    def visit_ptxExpression(self, node) -> (str, OpalType):  # Variable name, type name
        if isinstance(node, ast.BinOp):
            return self.visit_ptxBinOp(node)

        elif isinstance(node, ast.BoolOp):
            return self.visit_ptxBoolOp(node)

        elif isinstance(node, ast.UnaryOp):
            return self.visit_ptxUnaryOp(node)

        elif isinstance(node, ast.Constant):
            return self.visit_ptxConstant(node)

        elif isinstance(node, ast.Compare):
            return self.visit_ptxCompare(node)

        elif isinstance(node, ast.Call):
            return self.visit_ptxVariableReference(node)

        elif isinstance(node, ast.Name):
            return node.id, self.opal_variables[node.id]

        elif isinstance(node, ast.Subscript):
            return self.visit_ptxSubscript(node)

    def visit_Expr(self, node: ast.Expr):
        node.value = self.visit(node.value)
        if node.value is None:
            return None
        return node

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Attribute):
            return node

        def _root(node):
            if isinstance(node, ast.Name):
                return node
            if isinstance(node, ast.Attribute):
                return _root(node.value)
            return node

        root = _root(node.func)
        if not isinstance(root, ast.Name):
            return node
        if root.id != "ptx":
            return node

        parts = []

        _node = node.func
        while True:
            if isinstance(_node, ast.Name):
                break
            elif isinstance(_node, ast.Attribute):
                attr = _node.attr
                if attr == "_global":
                    attr = "global"
                parts = [attr] + parts
                _node = _node.value
        ptx_instruction = ".".join(parts) + " "
        instruction_parts = [ptx_instruction]
        for argument in node.args:
            if len(instruction_parts) != 1:
                instruction_parts.append(" , ")
            if isinstance(argument, ast.Set):
                exp = ast.Call(
                    func=ast.Attribute(
                        value=ast.Constant(value=","), attr="join", ctx=ast.Load()
                    ),
                    args=[ast.List(elts=argument.elts, ctx=ast.Load())],
                    keywords=[],
                )
                instruction_parts.append("{")
                instruction_parts.append(exp)
                instruction_parts.append("}")
            elif isinstance(argument, ast.List):
                instruction_parts.append("[")
                assert len(argument.elts) == 1
                instruction_parts.append(argument.elts[0])
                instruction_parts.append("]")
            else:
                instruction_parts.append(argument)

                # Array addressing thing
        self._insert_ptx_instruction(instruction_parts)
        return None

    def visit_For(self, node):
        # TODO: Opal For loops
        self._visit_body(node, "body")
        self._visit_body(node, "orelse")
        return node

    def visit_While(self, node):
        if not self.analyser.is_opal_expression(node.test, self.opal_variables):
            self._visit_body(node, "body")
            self._visit_body(node, "orelse")
            return node

        loop_header = self._new_block("loop_header")
        loop_body = self._new_block("loop_body")
        loop_exit = self._new_block("loop_exit")

        loop_header_body = []
        self.body_stack.append(loop_header_body)

        cond, typ = self.visit_ptxExpression(node.test)
        cond, typ = self.ptx_cast(typ, BasicType("pred"), cond)

        self._insert_ptx_instruction(
            [
                "@!",
                ast.Name(id=cond, ctx=ast.Load()),
                " bra $",
                ast.Name(id=loop_exit, ctx=ast.Load()),
            ]
        )
        self.body_stack.pop()

        self.push_expr(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(
                            id=loop_header,
                            ctx=ast.Load(),
                        )
                    )
                ],
                body=loop_header_body,
            ),
        )

        self._visit_body(node, "body")

        # this is most likely broken in one way.
        loop_body_postfix = []
        self.body_stack.append(loop_body_postfix)

        self._insert_ptx_instruction(
            [
                "bra $",
                ast.Name(id=loop_header, ctx=ast.Load()),
            ]
        )
        self.body_stack.pop()

        if isinstance(node.body[-1], ast.With):
            node.body[-1].body += loop_body_postfix
        else:
            node.body += loop_body_postfix

        self.push_expr(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(
                            id=loop_body,
                            ctx=ast.Load(),
                        )
                    )
                ],
                body=node.body,
            ),
        )

        self.push_body_continuation(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(
                            id=loop_exit,
                            ctx=ast.Load(),
                        )
                    )
                ],
                body=[],
            ),
            "body",
        )
        self.push_expr(ast.Pass())

    def _visit_body(self, node, body_attr):
        """
        Visit/Replace nodes of a body while allowing insertion of expressions.
        """
        new_body = []
        self.body_stack.append(new_body)
        body = getattr(node, body_attr)
        for i, cnode in enumerate(body):
            replace = self.visit(cnode)
            if replace is not None:
                self.body_stack[-1].append(replace)
        setattr(node, body_attr, new_body)
        self.body_stack.pop()

    def visit_If(self, node):
        if not self.analyser.is_opal_expression(node.test, self.opal_variables):
            self._visit_body(node, "body")
            self._visit_body(node, "orelse")
            return node

        cond, typ = self.visit_ptxExpression(node.test)
        cond, typ = self.ptx_cast(typ, BasicType("pred"), cond)

        if_block = self._new_block("if_cond")
        cond_not = self._new_block("not_cond")
        if_done = self._new_block("if_done") if len(node.orelse) > 0 else None
        self._insert_ptx_instruction(
            [
                "@!",
                ast.Name(id=cond, ctx=ast.Load()),
                " bra $",
                ast.Name(id=cond_not, ctx=ast.Load()),
            ]
        )

        self._visit_body(node, "body")

        if_postfix = []

        if len(node.orelse) > 0:
            self.body_stack.append(if_postfix)
            self._insert_ptx_instruction(
                ["bra $", ast.Name(id=if_done, ctx=ast.Load())]
            )
            self.body_stack.pop()

        if isinstance(node.body[-1], ast.With):
            node.body[-1].body += if_postfix
        else:
            node.body += if_postfix

        self.push_expr(
            ast.With(
                items=[
                    ast.withitem(context_expr=ast.Name(id=if_block, ctx=ast.Load()))
                ],
                body=node.body,
            )
        )
        if len(node.orelse) > 0:
            self._visit_body(node, "orelse")
            self.push_expr(
                ast.With(
                    items=[
                        ast.withitem(context_expr=ast.Name(id=cond_not, ctx=ast.Load()))
                    ],
                    body=node.orelse,
                )
            )

        self.push_body_continuation(
            ast.With(
                items=[
                    ast.withitem(
                        context_expr=ast.Name(
                            id=cond_not if len(node.orelse) == 0 else if_done,
                            ctx=ast.Load(),
                        )
                    )
                ],
                body=[],
            ),
            "body",
        )
        self.push_expr(ast.Pass())

    def visit_Assign(self, node):
        if len(node.targets) != 1:
            return node

        # FIXME: Register vector assignments.
        assert isinstance(node.targets[0], ast.Name)
        name = node.targets[0].id
        if name not in self.opal_variables:
            return node

        value, val_typ = self.visit_ptxExpression(node.value)
        target_type = self.opal_variables[name].get_fundamental_type()

        if val_typ.get_fundamental_type() != target_type:
            value, val_typ = self.ptx_cast(val_typ, self.opal_variables[name], value)

        self._insert_ptx_instruction(
            [
                f"mov.{target_type} ",
                ast.Name(id=name, ctx=ast.Load()),
                ", ",
                ast.Name(id=value, ctx=ast.Load()),
            ]
        )

    def visit_AugAssign(self, node):
        if (
            not isinstance(node.target, ast.Name)
            or node.target.id not in self.opal_variables
        ):
            return self.generic_visit(node)

        op = type(node.op).__name__
        op_type = self.opal_variables[node.target.id]
        right = self.visit_ptxExpression(node.value)
        right = self.ptx_cast(right[1], op_type, right[0])

        self._ptx_binop(
            op,
            op_type,
            node.target.id,
            (node.target.id, op_type),
            right,
        )

    def visit_AnnAssign(self, node):
        if not isinstance(node.target, ast.Name):
            return node

        typ = self.resolve_type(node.annotation)
        if not typ:
            return node

        name = node.target.id

        self.opal_variables[name] = typ
        assignment_expression = ast.Assign(
            targets=[node.target], value=self._new_variable(name, typ)
        )

        self.push_expr(assignment_expression)

        if not node.value:
            return

        value, val_typ = self.visit_ptxExpression(node.value)
        # print(value, typ, val_typ)
        value, typ = self.ptx_cast(val_typ, typ, value)
        self._insert_ptx_instruction(
            [
                f"mov.{typ.get_fundamental_type()} ",
                ast.Name(id=node.target.id, ctx=ast.Load()),
                ", ",
                ast.Name(id=value, ctx=ast.Load()),
            ]
        )

        # return assignment_expression

    def resolve_type(self, anno):
        if isinstance(anno, ast.Attribute):
            if not isinstance(anno.value, ast.Name) or anno.value.id == "BasicTypes":
                return None
            return BasicType(anno.attr)

        elif isinstance(anno, ast.Call):
            if not isinstance(anno.func, ast.Name):
                return None

            name = anno.func.id
            if name == "shared":
                # shared memory decleration.
                assert len(anno.args) == 2
                assert isinstance(anno.args[0], ast.Name)
                assert isinstance(anno.args[1], ast.Constant)

                return SharedMemoryType(anno.args[0].id, anno.args[1].value)

            elif name in TYPE_TO_REG:
                assert len(anno.args) == 1
                assert isinstance(anno.args[0], ast.Constant)

                return RegisterVector(name, anno.args[0].value)

        elif isinstance(anno, ast.Constant):
            return BasicType(anno.value)

        elif isinstance(anno, ast.Name):
            return BasicType(anno.id)

        return None

    def visit_FunctionDef(self, node):
        node.decorator_list = []

        # Setup temporary body for argument setup.
        self.body_stack.append([])

        if not self.is_fragment:
            kernel_arguments_st = []

            meta_arguments = []
            for arg in node.args.args:
                name = arg.arg

                typ = None
                if arg.annotation:
                    typ = self.resolve_type(arg.annotation)
                if not typ:
                    meta_arguments.append(arg)
                    continue

                self.opal_variables[name] = typ

                self.body_stack[-1].append(
                    ast.Assign(
                        targets=[ast.Name(id=name, ctx=ast.Store())],
                        value=self._new_variable(name, typ),
                    )
                )

                self._insert_ptx_instruction(
                    [
                        f"ld.param.{typ.get_fundamental_type()} ",
                        ast.Name(id=name, ctx=ast.Load()),
                        f", [{name}]",
                    ]
                )

                kernel_arguments_st.append(
                    ast.Tuple(
                        elts=[
                            ast.Constant(value=name),
                            ast.Constant(value=typ.type_name),
                        ],
                        ctx=ast.Load(),
                    )
                )
            self.push_expr(
                ast.Expr(
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id="kernel_builder", ctx=ast.Load()),
                            attr="generate_header",
                            ctx=ast.Load(),
                        ),
                        args=[
                            ast.Constant(value=node.name),
                            ast.List(elts=kernel_arguments_st, ctx=ast.Load()),
                        ],
                        keywords=[],
                    )
                )
            )

        arg_setup = self.body_stack[-1]
        self.body_stack = self.body_stack[:-1]
        self._visit_body(node, "body")

        node.body = arg_setup + node.body

        node.args.args = [
            ast.arg(
                arg="kernel_builder", lineno=node.lineno, col_offset=node.col_offset
            ),
        ] + meta_arguments

        return node


def kernel():
    """
    The opal kernel decorator.
    """

    def deco(func):
        function = ast.parse(inspect.getsource(func))
        visitor = OpalTransformer()
        function = visitor.visit(function)
        function = ast.fix_missing_locations(function)

        print("Unparsed\n", ast.unparse(function))
        transformed_code = compile(function, filename="<ast>", mode="exec")
        local_namespace = {}
        exec(transformed_code, func.__globals__, local_namespace)
        return local_namespace[func.__name__]

    return deco


def fragment():
    def deco(func):
        function = ast.parse(inspect.getsource(func))
        visitor = OpalTransformer(is_fragment=True)
        function = visitor.visit(function)
        function = ast.fix_missing_locations(function)
        # print("Unparsed\n", ast.unparse(function))
        transformed_code = compile(function, filename="<ast>", mode="exec")
        local_namespace = {}
        exec(transformed_code, func.__globals__, local_namespace)
        return local_namespace[func.__name__]

    return deco
