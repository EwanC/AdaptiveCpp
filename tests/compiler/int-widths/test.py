# SPDX-License-Identifier: BSD-2-Clause
"""Verifier, rejection, structural and executable integer legalization tests."""
import os
import random
import re
import subprocess
import sys

driver, tools, work = sys.argv[1:]


def lower(name, text):
    source = os.path.join(work, name + ".ll")
    output = os.path.join(work, name + ".legal.ll")
    with open(source, "w") as f:
        f.write(text)
    env = dict(os.environ, ACPP_DEBUG_LEVEL="0")
    result = subprocess.run([driver, source], env=env, text=True,
                            capture_output=True, check=True).stdout
    with open(output, "w") as f:
        f.write(result)
    subprocess.run([os.path.join(tools, "opt"), "-passes=verify",
                    "-disable-output", output], check=True)
    return result, output


def body(text, name):
    return re.search(r"define [^\n]*@" + name + r"\(.*?\n}", text,
                     re.S).group()


structural = r"""
target datalayout = "e-p:64:64-p3:64:64-i64:64-i128:128"
define <2 x i64> @initial_copyload(ptr %0, ptr %1, ptr addrspace(1) %2, ptr %dst) {
  %.sroa.016.0.copyload.i.i = load i128, ptr addrspace(1) %2, align 32, !tbaa !6, !alias.scope !3, !noalias !9
  store i128 %.sroa.016.0.copyload.i.i, ptr %dst, align 32, !tbaa !6, !alias.scope !3, !noalias !9
  %vector = bitcast i128 %.sroa.016.0.copyload.i.i to <2 x i64>
  ret <2 x i64> %vector
}
define void @copy(ptr addrspace(3) %src, ptr addrspace(3) %dst) {
  %x = load i128, ptr addrspace(3) %src, align 16, !range !0, !alias.scope !3
  store i128 %x, ptr addrspace(3) %dst, align 16, !alias.scope !3
  ret void
}
define void @loop(ptr %src, ptr %dst, i1 %again) {
entry:
  %v = load i128, ptr %src, align 1
  br label %loop
loop:
  %p = phi i128 [ %v, %entry ], [ %s, %loop ]
  %x = xor i128 %p, -1
  %s = select i1 %again, i128 %x, i128 0
  br i1 %again, label %loop, label %exit
exit:
  store i128 %s, ptr %dst, align 1
  ret void
}
define void @vectors(ptr %src, ptr %dst) {
  %v = load <4 x float>, ptr %src
  %x = bitcast <4 x float> %v to i128
  %r = xor i128 %x, 18446744073709551616
  %out = bitcast i128 %r to <4 x float>
  store <4 x float> %out, ptr %dst
  ret void
}
define void @narrow(ptr %src, ptr %dst) {
  %v = load i48, ptr %src, align 1
  %x = sext i48 %v to i128
  %s = ashr i128 %x, 65
  %t = trunc i128 %s to i48
  store i48 %t, ptr %dst, align 1
  ret void
}
define void @constants(ptr %dst, i1 %cond) {
  %v = select i1 %cond, i128 poison, i128 undef
  store i128 %v, ptr %dst
  ret void
}
define void @invalid(ptr %dst) {
  %v = shl i128 1, 128
  store i128 %v, ptr %dst
  ret void
}
define void @invalid_high(ptr %dst) {
  %v = lshr i128 -1, 18446744073709551616
  store i128 %v, ptr %dst
  ret void
}
define void @flags(ptr %dst) {
  %a = shl nuw i128 -1, 1
  %b = shl nsw i128 170141183460469231731687303715884105727, 1
  %c = lshr exact i128 1, 1
  %d = ashr exact i128 -1, 1
  %ab = or i128 %a, %b
  %cd = or i128 %c, %d
  %v = or i128 %ab, %cd
  store i128 %v, ptr %dst
  ret void
}
define void @dynamic(ptr %src, ptr %amount, ptr %dst) {
  %v = load i128, ptr %src
  %n = load i128, ptr %amount
  %s = shl i128 %v, %n
  store i128 %s, ptr %dst
  ret void
}
define void @source_poison(ptr %dst) {
  %left = shl i128 poison, 64
  %low = trunc i128 %left to i64
  store i64 %low, ptr %dst
  %extended = zext i64 poison to i128
  %right = lshr i128 %extended, 64
  %high = trunc i128 %right to i64
  store i64 %high, ptr %dst
  ret void
}
!0 = !{i128 0, i128 100}
!1 = distinct !{!1}
!2 = distinct !{!2, !1}
!3 = !{!2}
!4 = !{!"Simple C++ TBAA"}
!5 = !{!"omnipotent char", !4, i64 0}
!6 = !{!5, !5, i64 0}
!7 = distinct !{!7, !1}
!9 = !{!7}
"""
result, _ = lower("structural", structural)
assert not re.search(r"\bi128\b(?!:)", result)
initial = body(result, "initial_copyload")
assert initial.count("load i64") == 2 and initial.count("store i64") == 2
assert re.search(r"load i64, ptr addrspace\(1\) %2, align 32", initial)
assert re.search(r"load i64, ptr addrspace\(1\) %\w+, align 8", initial)
assert initial.count("!tbaa") == 4
assert initial.count("!alias.scope") == 4
assert initial.count("!noalias") == 4
copy = body(result, "copy")
assert copy.count("load i64") == 2 and copy.count("store i64") == 2
assert copy.rfind("load i64") < copy.find("store i64")
assert "align 16" in copy and "align 8" in copy
assert "!alias.scope" in copy and "!range" not in copy
assert "addrspace(3)" in copy
assert body(result, "loop").count("phi i64") == 2
assert body(result, "invalid").count("store i64 poison") == 2
assert body(result, "invalid_high").count("store i64 poison") == 2
assert body(result, "flags").count("store i64 poison") == 2
dynamic = body(result, "dynamic")
assert re.search(r"icmp ult i64 %\w+, 128", dynamic)
assert re.search(r"icmp eq i64 %\w+, 0", dynamic)
assert dynamic.count("i64 poison") == 2
assert body(result, "source_poison").count("store i64 poison") == 2
be, _ = lower("big-endian", structural.replace('"e-p:', '"E-p:'))
assert not re.search(r"\bi128\b(?!:)", be)
assert re.search(r"extractelement <2 x i64> %\w+, i32 1", body(be, "vectors"))

# A rejected function must not have even its otherwise-supported copy changed.
for op in ["add", "sub", "mul", "udiv", "sdiv", "urem", "srem"]:
    text = f"""
define void @reject(ptr %src, ptr %dst) {{
  %v = load i128, ptr %src
  store i128 %v, ptr %dst
  %x = {op} i128 %v, 3
  store i128 %x, ptr %dst
  ret void
}}
"""
    rejected, _ = lower("reject-" + op, text)
    assert rejected.count("load i128") == 1
    assert rejected.count("store i128") == 2
    assert "i64" not in rejected
for access in ["load volatile i128, ptr %src",
               "load atomic i128, ptr %src monotonic, align 16"]:
    rejected, _ = lower("reject-memory", f"""
define void @reject(ptr %src, ptr %dst) {{
  %v = {access}
  store i128 %v, ptr %dst
  ret void
}}
""")
    assert "store i128" in rejected and "store i64" not in rejected
rejected, _ = lower("reject-signature", """
define i128 @reject(ptr %src, ptr %dst) {
  %v = load i128, ptr %src
  store i128 %v, ptr %dst
  ret i128 %v
}
""")
assert "load i128" in rejected and "store i128" in rejected
rejected, _ = lower("reject-argument", """
define void @reject(i128 %x, ptr %src, ptr %dst) {
  %v = load i128, ptr %src
  store i128 %v, ptr %dst
  store i128 %x, ptr %dst
  ret void
}
""")
assert "load i128" in rejected and rejected.count("store i128") == 2
rejected, _ = lower("reject-connected-i96", """
define void @reject(i64 %input, ptr %dst) {
  %narrow = zext i64 %input to i96
  store i96 %narrow, ptr %dst
  %wide = zext i96 %narrow to i128
  store i128 %wide, ptr %dst
  ret void
}
""")
assert "store i96" in rejected and "store i128" in rejected
assert "store i64" not in rejected
rejected, _ = lower("reject-mixed-i96", """
define void @reject(i64 %input, ptr %src, ptr %dst) {
  %narrow = zext i64 %input to i96
  store i96 %narrow, ptr %dst
  %wide = load i128, ptr %src
  store i128 %wide, ptr %dst
  ret void
}
""")
assert "store i96" in rejected and "store i128" in rejected
assert "store i64" not in rejected
for access in ["store volatile i128 %v, ptr %dst",
               "store atomic i128 %v, ptr %dst monotonic, align 16"]:
    rejected, _ = lower("reject-store", f"""
define void @reject(ptr %src, ptr %dst) {{
  %v = load i128, ptr %src
  {access}
  ret void
}}
""")
    assert "load i128" in rejected and "load i64" not in rejected
for instruction in ["call i128 @unknown()", "freeze i128 1",
                    "fptoui double 1.0 to i128"]:
    rejected, _ = lower("reject-other", f"""
declare i128 @unknown()
define void @reject(ptr %dst) {{
  %v = {instruction}
  store i128 %v, ptr %dst
  ret void
}}
""")
    assert "store i128" in rejected and "store i64" not in rejected
rejected, _ = lower("reject-vector", """
define void @reject(ptr %src, ptr %dst) {
  %v = load <1 x fp128>, ptr %src
  %x = bitcast <1 x fp128> %v to i128
  store i128 %x, ptr %dst
  ret void
}
""")
assert "store i128" in rejected and "store i64" not in rejected

legacy, _ = lower("legacy", """
define void @copy48(ptr %src, ptr %dst) {
  %v = load i48, ptr %src, align 1
  %x = add i48 %v, 1
  store i48 %x, ptr %dst, align 1
  ret void
}
define void @copy96(ptr %src, ptr %dst) {
  %v = load i96, ptr %src, align 1
  store i96 %v, ptr %dst, align 1
  ret void
}
define void @pack96(i32 %a, i64 %b, ptr %dst) {
  %x = zext i64 %b to i96
  %y = zext i32 %a to i96
  %s = shl i96 %y, 64
  %r = or i96 %x, %s
  store i96 %r, ptr %dst, align 1
  ret void
}
""")
assert not re.search(r"\bi(?:48|96)\b", legacy)

# Generate deterministic semantic oracles without relying on LLVM's i128
# lowering to calculate the expected result.
mask = (1 << 128) - 1
rng = random.Random(128)
values = [0, 1, mask, 1 << 127, (1 << 64) - 1, 1 << 64]
values += [rng.getrandbits(128) for _ in range(12)]
functions = []
checks = []
index = 0


def check(expr, expected, inputs=""):
    global index
    name = "case" + str(index)
    functions.append(f"""
define void @{name}(ptr %out) {{
{inputs}
  %r = {expr}
  store i128 %r, ptr %out, align 1
  ret void
}}
""")
    checks.append(f"""
  call void @{name}(ptr %out)
  %v{index} = load i128, ptr %out, align 1
  %ok{index} = icmp eq i128 %v{index}, {expected & mask}
  %all{index} = and i1 %{"true" if index == 0 else "all" + str(index - 1)}, %ok{index}
""".replace("%true", "true"))
    index += 1


for value in values:
    signed = value if value < (1 << 127) else value - (1 << 128)
    for amount in [0, 1, 31, 32, 63, 64, 65, 95, 127]:
        for op, expected in [("shl", value << amount),
                             ("lshr", value >> amount),
                             ("ashr", signed >> amount)]:
            # Loads stop IRBuilder from reducing all the shifts as constants.
            inputs = f"""
  %in = alloca i128, align 16
  %shift = alloca i128, align 16
  store i128 {value}, ptr %in
  store i128 {amount}, ptr %shift
  %v = load i128, ptr %in
  %n = load i128, ptr %shift
"""
            check(f"{op} i128 %v, %n", expected, inputs)
            check(f"{op} i128 %v, {amount}", expected, inputs)
    other = rng.getrandbits(128)
    inputs = f"""
  %in = alloca i128
  store i128 {value}, ptr %in
  %v = load i128, ptr %in
"""
    for op, expected in [("and", value & other), ("or", value | other),
                         ("xor", value ^ other)]:
        check(f"{op} i128 %v, {other}", expected, inputs)
    for narrow in [1, 8, 16, 32, 48, 63, 64]:
        low = value & ((1 << narrow) - 1)
        sign = low if low < (1 << (narrow - 1)) else low - (1 << narrow)
        for cast, expected in [("zext", low), ("sext", sign)]:
            check(f"{cast} i{narrow} %narrow to i128", expected,
                  inputs + f"  %narrow = trunc i128 %v to i{narrow}\n")
    for vector in ["<16 x i8>", "<8 x i16>", "<4 x i32>",
                   "<2 x i64>", "<4 x float>", "<2 x double>"]:
        check(f"bitcast {vector} %vec to i128", value,
              inputs + f"  %vec = bitcast i128 %v to {vector}\n")
    for flag, expected, operation in [
        ("nuw", (value & ((1 << 127) - 1)) << 1, "shl"),
        ("nsw", (value & ((1 << 126) - 1)) << 1, "shl"),
        ("exact", (value & ~1) >> 1, "lshr")]:
        source = value & (((1 << 127) - 1) if flag == "nuw"
                           else ((1 << 126) - 1)) if operation == "shl" else value & ~1
        check(f"{operation} {flag} i128 %v, 1", expected,
              inputs.replace(f"store i128 {value},", f"store i128 {source},"))
    high = value & ~((1 << 64) - 1)
    # Exercise every predicate with identical values, equal high halves but
    # different low halves, and different high halves in both directions.
    for lhs, rhs in [(value, value), (value, other), (other, value),
                     (high, high | ((1 << 64) - 1)),
                     (high | ((1 << 64) - 1), high)]:
        lhs_signed = lhs if lhs < (1 << 127) else lhs - (1 << 128)
        rhs_signed = rhs if rhs < (1 << 127) else rhs - (1 << 128)
        for pred, expected in [
            ("eq", lhs == rhs), ("ne", lhs != rhs),
            ("ult", lhs < rhs), ("ule", lhs <= rhs),
            ("ugt", lhs > rhs), ("uge", lhs >= rhs),
            ("slt", lhs_signed < rhs_signed), ("sle", lhs_signed <= rhs_signed),
            ("sgt", lhs_signed > rhs_signed), ("sge", lhs_signed >= rhs_signed)]:
            check("zext i1 %c to i128", int(expected),
                  inputs.replace(f"store i128 {value},", f"store i128 {lhs},") +
                  f"  %c = icmp {pred} i128 %v, {rhs}\n")

check("bitcast <4 x i32> %vec to i128",
      1 | (2 << 32) | (3 << 64) | (4 << 96), """
  %slot = alloca <4 x i32>
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %slot
  %vec = load <4 x i32>, ptr %slot
""")
check("zext i32 %lane to i128", 3, """
  %slot = alloca i128
  store i128 316912650112397582603894390785, ptr %slot
  %v = load i128, ptr %slot
  %vec = bitcast i128 %v to <4 x i32>
  %lane = extractelement <4 x i32> %vec, i32 2
""")

# An overlapping copy must retain load-all-before-store behavior.
functions.append("""
define void @overlap(ptr %p) {
  %q = getelementptr i8, ptr %p, i64 8
  %v = load i128, ptr %p, align 1
  store i128 %v, ptr %q, align 1
  ret void
}
define void @phi_loop(ptr %out) {
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %x = phi i128 [ 1, %entry ], [ %bits, %loop ]
  %y = phi i48 [ 1, %entry ], [ %narrow, %loop ]
  %shift = shl i128 %x, 1
  %bits = or i128 %shift, 1
  %narrow = trunc i128 %bits to i48
  %wide = zext i48 %y to i128
  %result = select i1 true, i128 %bits, i128 %wide
  %next = add i32 %i, 1
  %done = icmp eq i32 %next, 7
  br i1 %done, label %exit, label %loop
exit:
  store i128 %result, ptr %out
  ret void
}
""")
program = "\n".join(functions) + """
define i32 @main() {
  %out = alloca i128, align 16
""" + "\n".join(checks) + f"""
  %buf = alloca [24 x i8], align 16
  store i128 36893488147419103233, ptr %buf
  call void @overlap(ptr %buf)
  %q = getelementptr i8, ptr %buf, i64 8
  %copied = load i128, ptr %q, align 1
  %copyok = icmp eq i128 %copied, 36893488147419103233
  call void @phi_loop(ptr %out)
  %loopresult = load i128, ptr %out
  %loopok = icmp eq i128 %loopresult, 255
  %extrachecks = and i1 %copyok, %loopok
  %ok = and i1 %all{index - 1}, %extrachecks
  %bad = xor i1 %ok, true
  %result = zext i1 %bad to i32
  ret i32 %result
}}
"""
lowered, output = lower("semantic", program)
assert not re.search(r"\bi128\b(?!:)", lowered)
subprocess.run([os.path.join(tools, "lli"), output], check=True)
print(f"Verified structural/rejection cases and {index} executable semantic cases")
