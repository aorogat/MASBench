from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from numpy.f2py.crackfortran import verbose


# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Datascientisttest():
    """Datascientisttest crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    #@agent
    #def researcher(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config['researcher'], # type: ignore[index]
       #     verbose=True
        #)

    #@agent
    #def reporting_analyst(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config['reporting_analyst'], # type: ignore[index]
       #     verbose=True
        #)
    #@agent
    #def data_scientist(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["data_scientist"],
       #     verbose=True
        #)
    @agent
    def no_role(self) -> Agent:
        return Agent(
            config=self.agents_config["no_role"],
            verbose=True
        )
    #@agent
    #def researcher(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["researcher"],
       #     verbose=True
        #)

    #@agent
    #def data_analyst(self) -> Agent:
     #  return Agent(
      #      config=self.agents_config["data_analyst"],
       #     verbose=True,
        #)
    #@agent
    #def engineer(self) -> Agent:
     #   return Agent(
      #      config=self.agents_config["engineer"],
       #     verbose=True,
        #)


    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
   #@task
    #def research_task(self) -> Task:
        #return Task(
        #    config=self.tasks_config['research_task'], # type: ignore[index]
        #)

    #@task
    #def reporting_task(self) -> Task:
        #return Task(
        #    config=self.tasks_config['reporting_task'], # type: ignore[index]
         #   output_file='report.md'
        #)
    @task
    def write_paragraph(self) -> Task:
        return Task(
            config=self.tasks_config["write_pipeline"],
            output_file='Wifi_no_role1.py',
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Datascientisttest crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
