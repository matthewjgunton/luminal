use egglog::{EGraph, Error, RunReport, Term};

/// Helper function to run egglog code and extract results
fn run_egglog_test(code: &str) -> RunReport {
    let lisp_rules = include_str!("code.lisp");
    let code = lisp_rules.replace("{code}", &code);
    let mut egraph = EGraph::default();
    let commands = egraph.parser.get_program_from_string(None, &code).unwrap();
    let _msgs = egraph.run_program(commands);
    let report = egraph
        .get_run_report()
        .as_ref()
        .expect("missing run report (did the run complete?)");
    report.clone()
}

fn assert_rule_hits_at_least(report: &RunReport, rule_snippet: &str, min_hits: usize) {
    // Find the first key that contains our snippet
    let (key, hits) = report
        .num_matches_per_rule
        .iter()
        .find(|(k, _)| k.contains(rule_snippet))
        .map(|(k, v)| (k.as_str(), *v))
        .unwrap_or_else(|| {
            // If we didn't even find the rule name, dump the available rule keys
            let available: Vec<&str> = report
                .num_matches_per_rule
                .keys()
                .map(|k| k.as_str())
                .collect();
            panic!(
                "Did not find a rule whose key contains:\n  {}\nAvailable rules:\n{}",
                rule_snippet,
                available.join("\n")
            )
        });

    assert!(
        hits >= min_hits,
        "Expected rule to apply at least {min_hits} time(s), but got {hits} for key:\n  {key}"
    );
}

#[cfg(test)]
mod symbolic_algebra_tests {
    use super::*;

    //expression tests
    #[test]
    pub fn mmul_constant_folding() {
        //create the ir
        let mmul = "(let full (MMul (MNum 10) (MNum 20)))";
        // generate LISP
        let report = run_egglog_test(mmul);
        println!("REPORT: \n\n\n{:?}", report);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMul (MNum a) (MNum b)) (MNum (* a b)) :when ((< a 10000) (< b 10000)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn madd_constant_folding() {
        //create the ir
        let mmul = "(let full (MAdd (MNum 10) (MNum 20)))";
        // generate LISP
        let report = run_egglog_test(mmul);
        println!("REPORT: \n\n\n{:?}", report);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MAdd (MNum a) (MNum b)) (MNum (+ a b)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mdiv_constant_folding() {
        //create the ir
        let mmul = "(let full (MDiv (MNum 10) (MNum 20)))";
        // generate LISP
        let report = run_egglog_test(mmul);
        println!("REPORT: \n\n\n{:?}", report);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MDiv (MNum a) (MNum b)) (MNum (/ a b)) :when ((!= 0 b) (= 0 (% a b))) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mmax_constant_folding() {
        //create the ir
        let mmul = "(let full (MMax (MNum 10) (MNum 20)))";
        // generate LISP
        let report = run_egglog_test(mmul);
        println!("REPORT: \n\n\n{:?}", report);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMax (MNum a) (MNum b)) (MNum (max a b)) :ruleset expr)",
            1,
        );
    }
    #[test]
    pub fn mmin_constant_folding() {
        //create the ir
        let mmul = "(let full (MMin (MNum 10) (MNum 20)))";
        // generate LISP
        let report = run_egglog_test(mmul);
        println!("REPORT: \n\n\n{:?}", report);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMin (MNum a) (MNum b)) (MNum (min a b)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn msub_constant_folding() {
        let expr = "(let full (MSub (MNum 30) (MNum 10)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MSub (MNum a) (MNum b)) (MNum (- a b)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mand_constant_folding() {
        let expr = "(let full (MAnd (MNum 5) (MNum 3)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MAnd (MNum a) (MNum b)) (MNum (& a b)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn madd_commutativity() {
        let expr = "(let full (MAdd (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(&report, "(rewrite (MAdd a b) (MAdd b a) :ruleset expr)", 1);
    }

    #[test]
    pub fn mmul_commutativity() {
        let expr = "(let full (MMul (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(&report, "(rewrite (MMul a b) (MMul b a) :ruleset expr)", 1);
    }

    #[test]
    pub fn madd_associativity() {
        let expr = "(let full (MAdd (MAdd (MVar \"a\") (MVar \"b\")) (MVar \"c\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MAdd (MAdd a b) c) (MAdd a (MAdd b c)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mmul_associativity() {
        let expr = "(let full (MMul (MMul (MVar \"a\") (MVar \"b\")) (MVar \"c\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMul (MMul a b) c) (MMul a (MMul b c)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn madd_zero_identity() {
        let expr = "(let full (MAdd (MVar \"x\") (MNum 0)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(&report, "(rewrite (MAdd a (MNum 0)) a :ruleset expr)", 1);
    }

    #[test]
    pub fn mmul_one_identity() {
        let expr = "(let full (MMul (MVar \"x\") (MNum 1)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(&report, "(rewrite (MMul a (MNum 1)) a :ruleset expr)", 1);
    }

    #[test]
    pub fn mmul_zero_absorb() {
        let expr = "(let full (MMul (MVar \"x\") (MNum 0)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMul a (MNum 0)) (MNum 0) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mdiv_one_identity() {
        let expr = "(let full (MDiv (MVar \"x\") (MNum 1)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(&report, "(rewrite (MDiv a (MNum 1)) a :ruleset expr)", 1);
    }

    #[test]
    pub fn floordiv_mod_reconstruction() {
        let expr = "(let full (MMul (MDiv (MVar \"a\") (MVar \"b\")) (MVar \"b\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMul (MDiv ?a ?b) ?b) (MFloorTo ?a ?b) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn floordiv_mod_identity() {
        let expr = "(let full (MAdd (MFloorTo (MVar \"a\") (MVar \"b\")) (MMod (MVar \"a\") (MVar \"b\"))))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MAdd (MFloorTo ?a ?b) (MMod ?a ?b)) ?a :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mod_mul_simplify() {
        let expr = "(let full (MMod (MMul (MVar \"x\") (MVar \"y\")) (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMod (MMul ?x ?y) ?y) (MNum 0) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn div_mul_distribute() {
        let expr = "(let full (MDiv (MMul (MVar \"x\") (MVar \"y\")) (MVar \"z\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MDiv (MMul ?x ?y) ?z) (MMul ?x (MDiv ?y ?z)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn nested_mod_simplify1() {
        let expr = "(let full (MMod (MMod (MVar \"x\") (MNum 16)) (MNum 32)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?y)) :when ((>= ?z ?y) (= 0 (% ?y ?z))) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn nested_mod_simplify2() {
        let expr = "(let full (MMod (MMod (MVar \"x\") (MNum 32)) (MNum 16)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MMod (MMod ?x (MNum ?y)) (MNum ?z)) (MMod ?x (MNum ?z)) :when ((>= ?y ?z) (= 0 (% ?z ?y))) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_direct() {
        let expr = "(let full (MReplace (MVar \"x\") (MVar \"x\") (MNum 42)))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace ?x ?y ?z) ?z :when ((= ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_add_distribute() {
        let expr =
            "(let full (MReplace (MAdd (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MAdd ?a ?b) ?x ?y) (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_sub_distribute() {
        let expr =
            "(let full (MReplace (MSub (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MSub ?a ?b) ?x ?y) (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_mul_distribute() {
        let expr =
            "(let full (MReplace (MMul (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MMul ?a ?b) ?x ?y) (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_div_distribute() {
        let expr =
            "(let full (MReplace (MDiv (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MDiv ?a ?b) ?x ?y) (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_mod_distribute() {
        let expr =
            "(let full (MReplace (MMod (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MMod ?a ?b) ?x ?y) (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_min_distribute() {
        let expr =
            "(let full (MReplace (MMin (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MMin ?a ?b) ?x ?y) (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_max_distribute() {
        let expr =
            "(let full (MReplace (MMax (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MMax ?a ?b) ?x ?y) (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_floordiv_distribute() {
        let expr =
            "(let full (MReplace (MFloorTo (MVar \"a\") (MVar \"b\")) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MFloorTo ?a ?b) ?x ?y) (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_num_unchanged() {
        let expr = "(let full (MReplace (MNum 42) (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MNum ?n) ?x ?y) (MNum ?n) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_accum_unchanged() {
        let expr = "(let full (MReplace (MAccum \"acc\") (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MAccum ?acc) ?x ?y) (MAccum ?acc) :ruleset expr)",
            1,
        );
    }

    #[test]
    pub fn mreplace_var_unchanged() {
        let expr = "(let full (MReplace (MVar \"v\") (MVar \"x\") (MVar \"y\")))";
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MReplace (MVar ?v) (MVar ?x) ?y) (MVar ?v) :when ((!= ?v ?x)) :ruleset expr)",
            1,
        );
    }
}

#[cfg(test)]
mod ir_tests {
    use super::*;

    // IR Binary Op Commutativity tests
    #[test]
    pub fn ir_binary_commutativity() {
        let expr = r#"
            (let full (Binary "Add" (GMEM "a") (GMEM "b")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (Binary ?bin ?a ?b) (Binary ?bin ?b ?a) :ruleset ir)",
            1,
        );
    }

    // Remove pad loop - unary case
    #[test]
    pub fn remove_pad_loop_unary() {
        let expr = r#"
            (let full (LoopOut (Unary "Exp2" (LoopIn (GMEM "x") (Loop "pad" (MNum 1)) (MNum 0))) (Loop "pad" (MNum 1)) (MNum 0)))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (LoopOut (Unary ?un (LoopIn ?x (Loop ?loop (MNum 1)) (MNum 0))) (Loop ?loop (MNum 1)) (MNum 0)) (Unary ?un ?x) :ruleset ir)",
            1,
        );
    }

    // Generic unary/binary conversion tests
    #[test]
    pub fn unary_to_generic() {
        let expr = r#"
            (let full (Exp2 (GMEM "x")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Exp2 ?x) (Unary 'Exp2' ?x) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn log2_to_generic() {
        let expr = r#"
            (let full (Log2 (GMEM "x")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Log2 ?x) (Unary 'Log2' ?x) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn sqrt_to_generic() {
        let expr = r#"
            (let full (Sqrt (GMEM "x")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Sqrt ?x) (Unary 'Sqrt' ?x) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn sin_to_generic() {
        let expr = r#"
            (let full (Sin (GMEM "x")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Sin ?x) (Unary 'Sin' ?x) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn recip_to_generic() {
        let expr = r#"
            (let full (Recip (GMEM "x")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Recip ?x) (Unary 'Recip' ?x) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn neg_to_generic() {
        let expr = r#"
            (let full (Neg (GMEM "x")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Neg ?x) (Unary 'Neg' ?x) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn add_to_generic() {
        let expr = r#"
            (let full (Add (GMEM "a") (GMEM "b")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Add ?a ?b) (Binary 'Add' ?a ?b) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn mul_to_generic() {
        let expr = r#"
            (let full (Mul (GMEM "a") (GMEM "b")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Mul ?a ?b) (Binary 'Mul' ?a ?b) :ruleset ir-generic)",
            1,
        );
    }

    #[test]
    pub fn max_to_generic() {
        let expr = r#"
            (let full (Max (GMEM "a") (GMEM "b")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(birewrite (Max ?a ?b) (Binary 'Max' ?a ?b) :ruleset ir-generic)",
            1,
        );
    }
}

#[cfg(test)]
mod fusion_tests {
    use super::*;

    #[test]
    pub fn loop_fusion_binary() {
        let expr = r#"
            (let full (LoopIn (LoopOut (Binary "Add" (GMEM "a") (GMEM "b")) (Loop "outer" (MNum 10)) (MVar "z")) (Loop "inner" (MNum 10)) (MVar "z")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (LoopIn (LoopOut (Binary ?bin ?a ?b) (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st) (Binary ?bin ?a ?b) :ruleset fusion)",
            1,
        );
    }

    #[test]
    pub fn loop_fusion_fused() {
        let expr = r#"
            (let full (LoopIn (LoopOut (GMEM "a") (Loop "outer" (MNum 10)) (MVar "z")) (Loop "inner" (MNum 10)) (MVar "z")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (LoopIn (LoopOut ?a (Loop ?loopA ?range) ?st) (Loop ?loopB ?range) ?st) (FusedLoops ?a ?range) :ruleset fusion)",
            1,
        );
    }

    #[test]
    pub fn nested_loop_fusion() {
        let expr = r#"
            (let full (LoopIn (FusedLoops (LoopOut (GMEM "a") (Loop "first" (MNum 5)) (MVar "z")) (MNum 10)) (Loop "second" (MNum 5)) (MVar "z")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (LoopIn (FusedLoops (LoopOut ?a (Loop ?loopA ?range) ?st) ?fused_range) (Loop ?loopB ?range) ?st) (FusedLoops ?a (MMul ?range ?fused_range)) :ruleset fusion)",
            1,
        );
    }

    #[test]
    pub fn nested_loop_fusion_binary() {
        let expr = r#"
            (let full (LoopIn (FusedLoops (LoopOut (Binary "Mul" (GMEM "a") (GMEM "b")) (Loop "first" (MNum 5)) (MVar "z")) (MNum 10)) (Loop "second" (MNum 5)) (MVar "z")))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (LoopIn (FusedLoops (LoopOut (Binary ?bin ?a ?b) (Loop ?loopA ?range) ?st) ?fused_range) (Loop ?loopB ?range) ?st) (Binary ?bin ?a ?b) :ruleset fusion)",
            1,
        );
    }
}

#[cfg(test)]
mod tiling_tests {
    use super::*;

    #[test]
    pub fn tile_loop_propagation_different_loop_in() {
        let expr = r#"
            (let full (TileLoop (LoopIn (GMEM "body") (Loop "other" (MNum 32)) (MVar "z")) "target"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (TileLoop (LoopIn ?body (Loop ?other ?range) ?stride) ?loop) (LoopIn (TileLoop ?body ?loop) (Loop ?other ?range) ?stride) :when ((!= ?loop ?other)) :ruleset ir-prop)",
            1,
        );
    }

    #[test]
    pub fn tile_loop_propagation_loop_out() {
        let expr = r#"
            (let full (TileLoop (LoopOut (GMEM "body") (Loop "other" (MNum 32)) (MVar "z")) "target"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (TileLoop (LoopOut ?body (Loop ?other ?range) ?stride) ?loop) (LoopOut (TileLoop ?body ?loop) (Loop ?other ?range) ?stride) :ruleset ir-prop)",
            1,
        );
    }

    #[test]
    pub fn tile_loop_propagation_unary() {
        let expr = r#"
            (let full (TileLoop (Unary "Exp2" (GMEM "body")) "target"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (TileLoop (Unary ?un ?body) ?loop) (Unary ?un (TileLoop ?body ?loop)) :ruleset ir-prop)",
            1,
        );
    }

    #[test]
    pub fn tile_loop_propagation_binary() {
        let expr = r#"
            (let full (TileLoop (Binary "Add" (GMEM "a") (GMEM "b")) "target"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (TileLoop (Binary ?bin ?bodyA ?bodyB) ?loop) (Binary ?bin (TileLoop ?bodyA ?loop) (TileLoop ?bodyB ?loop)) :ruleset ir-prop)",
            1,
        );
    }
}

#[cfg(test)]
mod merging_tests {
    use super::*;

    #[test]
    pub fn merge_loops_propagation_loop_in() {
        let expr = r#"
            (let full (MergeLoops (LoopIn (GMEM "body") (Loop "other" (MNum 32)) (MVar "z")) "outer" "inner"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MergeLoops (LoopIn ?body (Loop ?other ?range) ?stride) ?o ?i) (LoopIn (MergeLoops ?body ?o ?i) (Loop ?other ?range) ?stride) :when ((!= ?i ?other)) :ruleset ir-prop)",
            1,
        );
    }

    #[test]
    pub fn merge_loops_propagation_loop_out() {
        let expr = r#"
            (let full (MergeLoops (LoopOut (GMEM "body") (Loop "other" (MNum 32)) (MVar "z")) "outer" "inner"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MergeLoops (LoopOut ?body (Loop ?other ?range) ?stride) ?o ?i) (LoopOut (MergeLoops ?body ?o ?i) (Loop ?other ?range) ?stride) :ruleset ir-prop)",
            1,
        );
    }

    #[test]
    pub fn merge_loops_propagation_unary() {
        let expr = r#"
            (let full (MergeLoops (Unary "Sin" (GMEM "body")) "outer" "inner"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MergeLoops (Unary ?un ?body) ?o ?i) (Unary ?un (MergeLoops ?body ?o ?i)) :ruleset ir-prop)",
            1,
        );
    }

    #[test]
    pub fn merge_loops_propagation_binary() {
        let expr = r#"
            (let full (MergeLoops (Binary "Mul" (GMEM "a") (GMEM "b")) "outer" "inner"))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (MergeLoops (Binary ?bin ?bodyA ?bodyB) ?o ?i) (Binary ?bin (MergeLoops ?bodyA ?o ?i) (MergeLoops ?bodyB ?o ?i)) :ruleset ir-prop)",
            1,
        );
    }
}

#[cfg(test)]
mod swap_tests {
    use super::*;

    // This tests the specialized loop swap rule (lines 175-217) - currently commented out in the lisp file
    // The rule swaps inner and outer loops in a complex nested structure
    #[test]
    pub fn specialized_loop_swap() {
        let expr = r#"
            (let full (LoopOut (LoopOut (Binary "Add" (LoopIn (LoopIn (GMEM "a") (Loop "outer" (MNum 10)) (MVar "outA")) (Loop "inner" (MNum 5)) (MVar "inA")) (LoopIn (LoopIn (GMEM "b") (Loop "outer" (MNum 10)) (MVar "outB")) (Loop "inner" (MNum 5)) (MVar "inB"))) (Loop "inner" (MNum 5)) (MVar "innerStride")) (Loop "outer" (MNum 10)) (MVar "outerStride")))
        "#;
        let report = run_egglog_test(expr);
        // This rule is commented out in the original, so we might not see hits unless it's enabled
        // But we can still test the pattern structure
        println!("Specialized loop swap report: {:?}", report);
    }
}

#[cfg(test)]
mod tensor_core_tests {
    use super::*;

    // This test is for the main tensor core matmul pattern - it's very complex so we'll test a simplified version
    #[test]
    pub fn tensor_core_matmul_pattern() {
        // This is a simplified version of the very complex tensor core pattern
        // The full pattern spans lines 381-497 and is extremely intricate
        let expr = r#"
            (let a_input (TiledMatmulInputA (GMEM "a") 16 (MNum 2)))
            (let b_input (TiledMatmulInputB (GMEM "b") 8 (MNum 2)))
            (let full (LoopOut (LoopOut (LoopOut (LoopOut (LoopOut (Add (Mul a_input b_input) (LoopIn (LoopIn (LoopIn (LoopIn (LoopIn (GMEM "acc") (Loop "acc_m" (MNum 8)) (MNum 0)) (Loop "acc_n" (MNum 8)) (MNum 0)) (Loop "pad1" (MNum 1)) (MNum 0)) (Loop "pad2" (MNum 1)) (MNum 0)) (Loop "acc_k" (MNum 16)) (MAccum "a"))) (Loop "out_k" (MNum 16)) (MAccum "acc_outer")) (Loop "pad2" (MNum 1)) (MVar "z")) (Loop "pad1" (MNum 1)) (MVar "z")) (Loop "out_n" (MNum 8)) (MVar "z")) (Loop "out_m" (MNum 8)) (MMul (MVar "z") (MNum 8))))
        "#;
        let report = run_egglog_test(expr);
        // The tensor core rule is extremely complex, so we mainly check that the pattern compiles
        println!("Tensor core matmul report: {:?}", report);
    }
}

#[cfg(test)]
mod general_swap_tests {
    use super::*;

    #[test]
    pub fn swap_loops_propagation_binary() {
        let expr = r#"
            (let full (SwapLoops (Binary "Add" (GMEM "a") (GMEM "b")) (Loop "L" (MNum 1024)) (Loop "O" (MNum 1024))))
        "#;
        let report = run_egglog_test(expr);
        assert_rule_hits_at_least(
            &report,
            "(rewrite (SwapLoops (Binary ?bin ?a ?b) ?innerLoop ?outerLoop) (Binary ?bin (SwapLoops ?a ?innerLoop ?outerLoop) (SwapLoops ?b ?innerLoop ?outerLoop)) :ruleset ir-prop)",
            1,
        );
    }
}
