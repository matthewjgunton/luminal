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
}
