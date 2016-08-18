require_relative 'neuron'

class NeuronLayer

  attr_accessor :neurons, :outputs, :layer_number

  def initialize(num)
    @neurons = Array.new()
    @outputs = Array.new()
    @layer_number = num
  end

  def run_layer(inputs)
    @outputs = neurons.map.with_index do |neuron, i|
      neuron.set_inputs(inputs)
      neuron.neuron_output
    end
  end

  def teach_layer(summed_reverse_activation)
    neurons.each.with_index do |neuron, i|
      neuron.track_new_weights(summed_reverse_activation[i])
    end
  end

  def update_layer
    neurons.map!(&:update_weights)
    self
  end

  def create_layer(num_neurons_in_layer, num_of_inputs_per_neuron)
    @neurons = num_neurons_in_layer.times.map do |index|
      Neuron.new(num_of_inputs_per_neuron)
    end
  end

end
