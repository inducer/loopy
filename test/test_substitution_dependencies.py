"""Public, non-expanding substitution-DAG dependency queries."""

import numpy as np
import pytest

import loopy as lp
from loopy.kernel.tools import (
    InstructionDependencyInfo,
    get_instruction_dependency_info,
)
from loopy.symbolic import (
    get_dependencies,
    get_reduction_inames,
    get_substitution_rule_dependencies,
    parse,
)


def _program():
    return lp.make_kernel(
        "{[i,j]: 0<=i,j<4}",
        """
        base := a[j] + p
        used(x) := x + base
        unused(x,y) := used(x)
        row(x) := sum(j, unused(x, missing))
        identity(x) := x
        index(x) := x + offset
        out[index(i)] = identity(out[i]) + row(b[i]) {id=result}
        """,
        [lp.GlobalArg("out", np.float64, shape=(8,)), "..."],
        lang_version=(2018, 2),
        silenced_warnings=["inferred_iname"],
    )


def test_public_rule_dependencies_match_expansion():
    kernel = _program().default_entrypoint
    dependencies = get_substitution_rule_dependencies(kernel.substitutions)
    assert dependencies["base"] == frozenset({"a", "j", "p"})
    assert dependencies["unused"] == frozenset({"a", "j", "p", "x"})
    assert dependencies["row"] == frozenset({"a", "p", "x"})
    assert all(isinstance(names, frozenset) for names in dependencies.values())
    assert dependencies.keys() == kernel.substitutions.keys()

    # Query the body, not a call: formal arguments must remain symbolic.
    for name, rule in kernel.substitutions.items():
        probe = kernel.copy(instructions=[
            lp.Assignment("probe", rule.expression, id="probe"),
        ])
        expanded = lp.expand_subst(probe).instructions[0].expression
        assert dependencies[name] == get_dependencies(expanded)


def test_public_instruction_dependencies_match_expansion():
    kernel = _program().default_entrypoint
    info = get_instruction_dependency_info(kernel)
    assert info["result"] == InstructionDependencyInfo(
        read_dependency_names=frozenset({"a", "b", "i", "offset", "out", "p"}),
        reduction_inames=frozenset({"j"}),
    )
    assert info.keys() == kernel.id_to_insn.keys()
    for insn in lp.expand_subst(kernel).instructions:
        assert info[insn.id].read_dependency_names == insn.read_dependency_names()
        assert info[insn.id].reduction_inames == insn.reduction_inames()


@pytest.mark.parametrize("swept", [False, True])
def test_public_instruction_dependencies_preserve_assignee_self_index(swept):
    rules = {"one": lp.SubstitutionRule("one", (), parse("1"))}
    if swept:
        instruction = lp.CallInstruction(
            (parse("[j]: out[out[0,0],j]"),),
            parse("f(a[i], one)"), id="result",
        )
        out_shape = (4, 4)
    else:
        instruction = lp.Assignment(
            parse("out[out[0]]"), parse("a[i] + one"), id="result",
        )
        out_shape = (4,)
    kernel = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}", [instruction], substitutions=rules,
        kernel_data=[
            lp.GlobalArg("a", np.int32, shape=(4,)),
            lp.GlobalArg("out", np.int32, shape=out_shape),
        ],
        lang_version=(2018, 2),
    ).default_entrypoint
    info = get_instruction_dependency_info(kernel)["result"]
    expanded, = lp.expand_subst(kernel).instructions
    assert info.read_dependency_names == expanded.read_dependency_names()
    assert info.read_dependency_names == frozenset({"a", "i", "out"})


def test_public_dependencies_without_rules():
    kernel = lp.make_kernel(
        "{[i,j]: 0<=i,j<4}", "out[i] = sum(j, a[i,j])",
        [
            lp.GlobalArg("a", np.float64, shape=(4, 4)),
            lp.GlobalArg("out", np.float64, shape=(4,), is_input=False),
        ],
        lang_version=(2018, 2),
    ).default_entrypoint
    assert get_substitution_rule_dependencies({}) == {}
    info = get_instruction_dependency_info(kernel)
    insn, = kernel.instructions
    assert info[insn.id].read_dependency_names == insn.read_dependency_names()
    assert info[insn.id].reduction_inames == get_reduction_inames(insn.expression)


def test_public_dependency_queries_do_not_expand(monkeypatch):
    import loopy.transform.subst

    rules = ["level_0(x) := x + 1"]
    for level in range(1, 19):
        rules.append(
            f"level_{level}(x) := "
            f"level_{level - 1}(x) + level_{level - 1}(x)"
        )
    program = lp.make_kernel(
        "{[i]: 0<=i<4}",
        [*rules, "out[i] = level_18(a[i]) {id=result}"],
        [
            lp.GlobalArg("a", np.float64, shape=(4,)),
            lp.GlobalArg("out", np.float64, shape=(4,), is_input=False),
        ],
        lang_version=(2018, 2),
    )

    def reject(*args, **kwargs):
        pytest.fail("dependency query expanded substitution rules")

    monkeypatch.setattr(lp, "expand_subst", reject)
    monkeypatch.setattr(loopy.transform.subst, "expand_subst", reject)
    kernel = program.default_entrypoint
    assert set(get_substitution_rule_dependencies(kernel.substitutions).values()) == {
        frozenset({"x"}),
    }
    assert get_instruction_dependency_info(kernel)["result"].read_dependency_names == {
        "a", "i",
    }


def test_public_rule_query_does_not_reuse_another_graph_cache():
    rules = {"f": lp.SubstitutionRule("f", (), parse("a[i]"))}
    assert get_substitution_rule_dependencies(rules) == {
        "f": frozenset({"a", "i"}),
    }
    rules["f"] = lp.SubstitutionRule("f", (), parse("b[j]"))
    assert get_substitution_rule_dependencies(rules) == {
        "f": frozenset({"b", "j"}),
    }


@pytest.mark.parametrize(("rules", "message"), [
    (
        {
            "f": lp.SubstitutionRule("f", (), parse("g")),
            "g": lp.SubstitutionRule("g", (), parse("f")),
        },
        "recursive substitution rules",
    ),
    (
        {
            "f": lp.SubstitutionRule("f", ("x",), parse("x")),
            "g": lp.SubstitutionRule("g", (), parse("f(a, b)")),
        },
        "number of arguments",
    ),
])
def test_public_rule_query_preserves_invalid_rule_errors(rules, message):
    with pytest.raises(lp.LoopyError, match=message):
        get_substitution_rule_dependencies(rules)
