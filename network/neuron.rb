require_relative 'synapse'

class Neuron

  attr_reader :bias, :synapses, :inputs, :output, :activation_error

  LR = 0.01

  def initialize(number_of_weights)
    @bias = -1
    @synapses = (number_of_weights + 1).times.map { Synapse.new(Random.rand(-0.5..0.5)) }
    @inputs = Array.new
  end

  def set_inputs(inputs)
    clear_inputs
    @inputs << bias
    @inputs << inputs
    @inputs.flatten!
  end

  def randomize_weights
    synapses.each do |synapse|
      synapse.weight = Random.rand(-0.5..0.5)
    end
  end

  def update_weights
    synapses.map!(&:update!)
    self
  end

  def neuron_output
    clear_output
    @output = sigmoid
    @output
  end

  def clear_output
    @output = nil
  end

  def clear_inputs
    @inputs = Array.new
  end

  def track_new_weights(summed_input_error)
    synapses.each.with_index do |synapse, index|
      error_rate = delta(inputs[index], summed_input_error)
      synapse.store_new_weight(error_rate)
    end
  end

  def delta(input, summed_input_error)
    derivative(summed_input_error) * LR * input
  end

  def derivative(summed_input_error)
    summed_input_error * (output * (1 - output))
  end

  private

  def sigmoid
    (1 / 1 + (1**-activation(inputs)))
  end

  def activation(input_vec)
    input_vec.map.with_index do |input, index|
      synapses[index].firing_rate(input)
    end.inject(:+)
  end

end
