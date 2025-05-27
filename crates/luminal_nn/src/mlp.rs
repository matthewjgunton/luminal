use luminal::prelude::*;

use crate::Linear;

pub struct Mlp {
    pub fc1: Linear, // hidden -> intermediate
    pub fc2: Linear, // intermediate -> hidden
}

impl Mlp {
    pub fn new(hidden: usize, intermediate: usize, cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new_permuted(hidden, intermediate, false, cx),
            fc2: Linear::new_permuted(intermediate, hidden, false, cx),
        }
    }
    pub fn new_with_bias(hidden: usize, intermediate: usize, cx: &mut Graph) -> Self {
        Self {
            fc1: Linear::new_permuted(hidden, intermediate, true, cx),
            fc2: Linear::new_permuted(intermediate, hidden, true, cx),
        }
    }
    pub fn new_different_intermediary(
        input: usize,
        intermediate: usize,
        output: usize,
        cx: &mut Graph,
    ) -> Self {
        Self {
            fc1: Linear::new_permuted(input, intermediate, true, cx),
            fc2: Linear::new_permuted(intermediate, output, true, cx),
        }
    }
}

impl Module<GraphTensor> for Mlp {
    type Output = GraphTensor;
    fn forward(&self, x: GraphTensor) -> Self::Output {
        self.fc2.forward(self.fc1.forward(x).gelu())
    }
}

impl SerializeModule for Mlp {
    fn serialize(&self, s: &mut Serializer) {
        s.module("fc1", &self.fc1);
        s.module("fc2", &self.fc2);
    }
}
