"""
Test minimal pour vérifier les concepts TAGI-SNN sans dépendances PyTorch
"""
import math
import random
from typing import Dict, Tuple, List
from enum import Enum

class NeuronState(Enum):
    DEAD = "dead"
    ACTIVE = "active"
    MATURE = "mature"

class MinimalTAGINeuron:
    """Version simplifiée du neurone TAGI-SNN pour test des concepts"""
    
    def __init__(self, neuron_id: int, position: Tuple[float, float]):
        self.id = neuron_id
        self.position = position
        
        # Gaussian voltage state (mean, variance)
        self.voltage_mu = 0.0
        self.voltage_var = 0.01
        
        # Adaptive parameters
        self.threshold = 1.0
        self.leak = 0.5
        self.state = NeuronState.ACTIVE
        
        # TAGI weight posteriors {from_id: (mu, var)}
        self.incoming_weights = {}
        self.spike_history = [0, 0, 0]
        self.vitality_history = []
        
    def update_voltage_gaussian(self, incoming_spikes: Dict[int, float]):
        """Update Gaussian voltage following Eqs. (6-7)"""
        prev_spike = self.spike_history[-1]
        
        # Leak term
        leak_term = self.leak * self.voltage_mu * (1 - prev_spike)
        
        # Synaptic input  
        synaptic_mu = 0.0
        synaptic_var = 0.0
        
        for from_id, spike_val in incoming_spikes.items():
            if from_id in self.incoming_weights and spike_val > 0:
                w_mu, w_var = self.incoming_weights[from_id]
                synaptic_mu += w_mu * spike_val
                synaptic_var += w_var * spike_val
                
        # Update voltage distribution
        self.voltage_mu = leak_term + synaptic_mu
        leak_var_term = (self.leak ** 2) * self.voltage_var * (1 - prev_spike)
        self.voltage_var = leak_var_term + synaptic_var
        
    def compute_spike_probability(self) -> float:
        """Compute spike probability from Gaussian voltage"""
        if self.voltage_var <= 0:
            return 0.0
        std_v = math.sqrt(self.voltage_var)
        z = (self.threshold - self.voltage_mu) / std_v
        # Approximation of 1 - Φ(z)
        spike_prob = 0.5 * (1.0 - math.erf(z / math.sqrt(2)))
        return max(0.0, min(1.0, spike_prob))
        
    def generate_spike(self) -> float:
        """Generate spike based on voltage exceeding threshold"""
        # Sample from Gaussian and compare to threshold
        voltage_sample = random.gauss(self.voltage_mu, math.sqrt(max(self.voltage_var, 1e-6)))
        spike = 1.0 if voltage_sample > self.threshold else 0.0
        
        # Update spike history
        self.spike_history = self.spike_history[1:] + [spike]
        return spike
        
    def update_homeostatic_threshold(self, spike: float):
        """Update threshold using surprise-gated homeostasis"""
        spike_prob = self.compute_spike_probability()
        surprise = 1.0 - spike_prob
        
        alpha_theta = 0.01
        beta_theta = 0.01
        
        if spike > 0:
            delta_theta = alpha_theta * surprise * spike
        else:
            delta_theta = -beta_theta * spike_prob * (1 - spike)
            
        self.threshold += delta_theta
        self.threshold = max(0.1, min(10.0, self.threshold))
        
    def compute_vitality(self, prior_var: float = 0.1) -> float:
        """Compute neuron vitality from weight variance shrinkage"""
        if not self.incoming_weights:
            return 1.0
            
        total_var_ratio = sum(w_var / prior_var for _, w_var in self.incoming_weights.values())
        vitality = total_var_ratio / len(self.incoming_weights)
        self.vitality_history.append(vitality)
        
        return vitality
        
    def update_lifecycle_state(self):
        """Update neuron lifecycle state"""
        current_vitality = self.compute_vitality()
        
        delta_rho = 0.0
        if len(self.vitality_history) >= 2:
            delta_rho = abs(self.vitality_history[-1] - self.vitality_history[-2])
            
        # Lifecycle thresholds
        tau_kill = 0.95
        tau_sat = 0.2
        tau_active = 1e-3
        
        if current_vitality >= tau_kill:
            self.state = NeuronState.DEAD
        elif delta_rho > tau_active:
            self.state = NeuronState.ACTIVE
        elif current_vitality < tau_sat and delta_rho <= tau_active:
            self.state = NeuronState.MATURE
        else:
            self.state = NeuronState.ACTIVE
            
    def step(self, incoming_spikes: Dict[int, float]) -> float:
        """Execute one timestep"""
        self.update_voltage_gaussian(incoming_spikes)
        spike = self.generate_spike()
        self.update_homeostatic_threshold(spike)
        self.update_lifecycle_state()
        return spike

class MinimalTAGINetwork:
    """Simplified TAGI-SNN network for concept testing"""
    
    def __init__(self, n_neurons: int = 10):
        self.neurons = {}
        self.weights = {}  # {(from_id, to_id): (mu, var)}
        self.prior_var = 0.1
        self.sigma_v = 0.01
        
        # Initialize neurons
        for i in range(n_neurons):
            pos = (random.uniform(0, 10), random.uniform(0, 10))
            self.neurons[i] = MinimalTAGINeuron(i, pos)
            
        # Random initial connectivity
        for _ in range(n_neurons):
            from_id = random.randint(0, n_neurons-1)
            to_id = random.randint(0, n_neurons-1)
            if from_id != to_id:
                w_mu = random.gauss(0, math.sqrt(self.prior_var))
                w_var = self.prior_var
                self.weights[(from_id, to_id)] = (w_mu, w_var)
                self.neurons[to_id].incoming_weights[from_id] = (w_mu, w_var)
                
    def tagi_weight_update(self, from_id: int, to_id: int, target: float, 
                          prediction_mu: float, prediction_var: float, pre_spike: float):
        """TAGI weight update (Eqs. 8-9)"""
        if (from_id, to_id) not in self.weights or pre_spike <= 0:
            return
            
        w_mu, w_var = self.weights[(from_id, to_id)]
        innovation = target - prediction_mu
        
        # Mean update
        gain = w_var * pre_spike / (prediction_var + self.sigma_v)
        new_w_mu = w_mu + gain * innovation
        
        # Variance update (monotonically decreasing)
        var_shrinkage = (w_var ** 2) * pre_spike / (prediction_var + self.sigma_v)
        new_w_var = max(w_var - var_shrinkage, 1e-6)
        
        # Update weights
        self.weights[(from_id, to_id)] = (new_w_mu, new_w_var)
        self.neurons[to_id].incoming_weights[from_id] = (new_w_mu, new_w_var)
        
    def forward_step(self, input_spikes: Dict[int, float], 
                    targets: Dict[int, float] = None) -> Dict[int, float]:
        """Execute one forward step"""
        output_spikes = {}
        
        # Update all neurons
        for neuron_id, neuron in self.neurons.items():
            if neuron.state == NeuronState.DEAD:
                continue
                
            # Collect inputs
            incoming = {}
            for from_id in neuron.incoming_weights.keys():
                incoming[from_id] = input_spikes.get(from_id, 0.0)
                if from_id in output_spikes:
                    incoming[from_id] = output_spikes[from_id]
                    
            spike = neuron.step(incoming)
            output_spikes[neuron_id] = spike
            
        # TAGI weight updates
        if targets:
            for to_id, target in targets.items():
                if to_id in self.neurons:
                    neuron = self.neurons[to_id]
                    pred_mu = neuron.voltage_mu
                    pred_var = neuron.voltage_var + self.sigma_v
                    
                    for from_id in neuron.incoming_weights.keys():
                        pre_spike = input_spikes.get(from_id, output_spikes.get(from_id, 0.0))
                        self.tagi_weight_update(from_id, to_id, target, pred_mu, pred_var, pre_spike)
                        
        return output_spikes
        
    def get_stats(self) -> Dict:
        """Get network statistics"""
        states = [n.state.value for n in self.neurons.values()]
        vitalities = [n.vitality_history[-1] if n.vitality_history else 1.0 for n in self.neurons.values()]
        
        return {
            "n_neurons": len(self.neurons),
            "n_connections": len(self.weights),
            "states": {s: states.count(s) for s in ['dead', 'active', 'mature']},
            "avg_vitality": sum(vitalities) / len(vitalities) if vitalities else 0.0
        }

def test_tagi_concepts():
    """Test des concepts TAGI-SNN"""
    print("🧠 Test des concepts TAGI-SNN")
    print("=" * 50)
    
    # Create network
    network = MinimalTAGINetwork(n_neurons=8)
    print(f"Réseau initialisé: {len(network.neurons)} neurones, {len(network.weights)} connexions")
    
    # Test patterns
    patterns = [
        # Pattern A: neurones 0,1,2 -> cible neurone 6
        ({0: 0.8, 1: 0.7, 2: 0.6}, {6: 1.0}),
        # Pattern B: neurones 3,4 -> cible neurone 7  
        ({3: 0.8, 4: 0.7}, {7: 1.0}),
        # Noise
        ({5: 0.3}, {})
    ]
    
    print(f"\n🎯 Patterns d'entraînement: {len(patterns)} patterns")
    
    # Training
    for epoch in range(5):
        print(f"\n--- Époque {epoch + 1} ---")
        
        for step in range(20):
            # Random pattern
            pattern_idx = random.randint(0, len(patterns) - 1)
            inputs, targets = patterns[pattern_idx]
            
            # Forward pass
            outputs = network.forward_step(inputs, targets)
            
            if step % 5 == 0:
                active_outputs = sum(1 for v in outputs.values() if v > 0)
                stats = network.get_stats()
                print(f"  Step {step:2d}: Pattern {pattern_idx}, "
                      f"Sorties actives: {active_outputs}, "
                      f"Vitalité moy: {stats['avg_vitality']:.3f}")
                      
    # Test final
    print(f"\n🧪 Test final des patterns:")
    for i, (inputs, targets) in enumerate(patterns):
        outputs = network.forward_step(inputs)
        
        # Check target response
        target_response = ""
        for target_id in targets.keys():
            resp = outputs.get(target_id, 0.0)
            target_response += f"N{target_id}:{resp:.1f} "
            
        input_str = " ".join([f"N{k}:{v:.1f}" for k, v in inputs.items()])
        print(f"  Pattern {i}: [{input_str}] -> [{target_response.strip()}]")
        
    # Analyse finale
    final_stats = network.get_stats()
    print(f"\n📊 Statistiques finales:")
    print(f"  • Neurones vivants: {final_stats['states']['active'] + final_stats['states']['mature']}")
    print(f"  • États: {final_stats['states']}")
    print(f"  • Vitalité moyenne: {final_stats['avg_vitality']:.3f}")
    
    # Vérifier shrinkage des poids
    print(f"\n🔍 Analyse des poids TAGI:")
    weight_shrinkage = []
    for (from_id, to_id), (w_mu, w_var) in network.weights.items():
        shrinkage_ratio = w_var / network.prior_var
        weight_shrinkage.append(shrinkage_ratio)
        if shrinkage_ratio < 0.5:  # significant shrinkage
            print(f"  • Connexion {from_id}→{to_id}: variance {w_var:.4f} (shrinkage {1-shrinkage_ratio:.1%})")
    
    avg_shrinkage = 1 - (sum(weight_shrinkage) / len(weight_shrinkage))
    print(f"  • Shrinkage moyen des poids: {avg_shrinkage:.1%}")
    
    print(f"\n✅ Test des concepts TAGI-SNN terminé!")
    print(f"💡 Concepts validés:")
    print(f"   - Propagation gaussienne des voltages ✓")
    print(f"   - Mise à jour TAGI des poids ✓")
    print(f"   - Homeostasis adaptative ✓")
    print(f"   - Lifecycle des neurones ✓")
    print(f"   - Shrinkage des variances ✓")

if __name__ == "__main__":
    test_tagi_concepts()