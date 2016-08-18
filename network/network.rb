require_relative 'neuron_layer'
require 'pry_debug'

class Network

  attr_accessor :layers, :output_layer

  def initialize(network_setup)
    @layers = layer_setup(network_setup)
    @output_layer = NeuronLayer.new(network_setup[1])
    @output_layer.create_layer(network_setup[3], network_setup[2].last)
  end

  def execute(input)
    run(input).last.max
  end

  def train(training_set)
    expected_answers = training_set.transpose.last
    count = 0

    while count < 20_000
      training_set.each.with_index do |set, index|
        ds = set[0]
        outputs = run(ds)
        ai_answers = outputs.last
        emr = start_backwards_propagate_inputs(ai_answers, expected_answers)
        output_layer.teach_layer(emr)
        t = back_propagate_data(emr, output_layer)
        output_layer.update_layer
        layers.map!(&:update_layer)
      end
      count += 1
    end

  end

  def start_backwards_propagate_inputs(ai_answers, expected_answers)
    ai_answers.map.with_index do |answer, i|
      answer * (1 - answer) * (expected_answers[i])
    end
  end

  def back_propagate_data(input_error_margins, input_layer)
    layers.map do |layer|
      neuron_input_weights = layer.neurons.map.with_index do |neuron, index|
        input_layer.neurons.map do |input_neuron|
          input_neuron.synapses[index].weight
        end
      end

      total_neurons_error_margins = neuron_input_weights.map do |weights|
        total = weights.map.with_index do |weight, index|
          weight * input_error_margins[index]
        end.inject(:+)
      end



      layer.teach_layer(total_neurons_error_margins)
      input_layer = layer

    end
  end

  def get_neurons_reverse_synapse_weights(current_layer, input_layer, synapse_index)
    current_layer.neurons.map.with_index do |neuron, i|
      input_layer.neurons.map do |input_neuron|
        input_neuron.synapses[synapse_index].weight
      end
    end
  end

  def run(inputs)
    outs = nil
    outputs = layers.map.with_index do |layer, index|
      inputs = outs if index > 0
      outs = layer.run_layer(inputs)
    end
    outputs << output_layer.run_layer(outs)
  end


  private
  def layer_setup(network_setup)
    network_setup[1].times.map do |index|
      layer = NeuronLayer.new(index)
      input_count = index > 0 ? network_setup[2][index-1] : network_setup[0]
      layer.create_layer(network_setup[2][index], input_count)
      layer
    end
  end

end
