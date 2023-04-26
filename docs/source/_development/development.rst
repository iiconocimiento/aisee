.. _development:

Contribute to AISee
===================

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
`code of conduct <https://github.com/iiconocimiento/aisee/blob/main/CODE_OF_CONDUCT.md>`_.

**This guide was heavily inspired by the awesome** 🤗 `Transformers guide to contributing <https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md>`_.

Ways to contribute
------------------

There are several ways you can contribute to AISee:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.


Fixing outstanding issues
-------------------------

If you notice an issue with the existing code and have a fix in mind,
feel free to `start contributing <https://github.com/iiconocimiento/aisee/blob/main/CONTRIBUTING.md/#create-a-pull-request>`_
and open a Pull Request!

Submitting a bug-related issue or feature request
-------------------------------------------------

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

Did you find a bug?
~~~~~~~~~~~~~~~~~~~

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues).

Once you've confirmed the bug hasn't already been reported, please include the following information
in your issue so we can quickly resolve it:

* Your **OS type and version**, **Python** and **PyTorch** versions when applicable.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help. Also explain what the expected behavior should be in your opinion.

Do you want a new feature?
~~~~~~~~~~~~~~~~~~~~~~~~~~

If there is a new feature you'd like to see in AISee, please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it a feature related to something you need for a project? Is it something you worked on and think it could benefit the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better we'll be able to help you.
3. Provide a *code snippet* that demonstrates the features usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

Do you want to add documentation?
---------------------------------

We're always looking for improvements to the documentation that make it more clear and accurate. Please
let us know how the documentation can be improved such as typos and any content that is missing, unclear
or inaccurate. We'll be happy to make the changes or help you make a contribution if you're interested!

Create a Pull Request
---------------------

Before writing any code, we strongly advise you to search through the existing PRs or
issues to make sure nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

Follow the steps below to start contributing:

1. Fork the `repository <https://github.com/iiconocimiento/aisee>`_ by
   clicking on the `Fork <https://github.com/iiconocimiento/aisee/fork>`_ button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote:

    .. code-block:: console

        git clone git@github.com:<your Github handle>/aisee.git
        cd aisee
        git remote add upstream https://github.com/iiconocimiento/aisee.git

3. Create a new branch to hold your development changes:

    .. code-block:: console

        git checkout -b a-descriptive-name-for-my-changes

   🚨 **Do not** work on the `main` branch!

4. Develop the features on your branch.

   As you work on your code, you should make sure the test suite
   passes. Run the tests impacted by your changes like this:

    .. code-block:: console

        poetry run pytest tests/<TEST_TO_RUN>.py

   AISee relies on `black` and `ruff` to format its source code
   consistently. After you make changes, apply automatic style corrections and code verifications
   that can't be automated in one go.

   If you're modifying documents under `doc` directory, make sure the documentation can still be built.
   This check will also run in the CI when you open a pull request.

   Once you're happy with your changes, add changed files with `git add` and
   record your changes locally with `git commit`:

    .. code-block:: console

        git add modified_file.py
        git commit

   Please remember to write `good commit messages <https://chris.beams.io/posts/git-commit/>`_ to clearly
   communicate the changes you made!

   To keep your copy of the code up to date with the original
   repository, rebase your branch on `upstream/branch` *before* you open a pull request or if requested by a maintainer:

    .. code-block:: console

        git fetch upstream
        git rebase upstream/main

   Push your changes to your branch:

    .. code-block:: console

        git push -u origin a-descriptive-name-for-my-changes

   If you've already opened a pull request, you'll need to force push with the `--force` flag.
   Otherwise, if the pull request hasn't been opened yet, you can just push your changes normally.

5. Now you can go to your fork of the repository on GitHub and click on **Pull request** to open a pull request. When you're ready, you can send your changes to the project maintainers for review.

6. It's ok if maintainers request changes, it happens to our core contributors
   too! So everyone can see the changes in the pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.

Pull request checklist
~~~~~~~~~~~~~~~~~~~~~~

☐ The pull request title should summarize your contribution.

☐ If your pull request addresses an issue, please mention the issue number in the pull
request description to make sure they are linked (and people viewing the issue know you
are working on it).

☐ To indicate a work in progress please prefix the title with `[WIP]` or `[DRAFT]`. These are
useful to avoid duplicated work, and to differentiate it from PRs ready to be merged.

☐ Make sure existing tests pass.

☐ If adding a new feature, also add tests for it.

☐ All public methods must have informative docstrings in `NumPy format <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`_.

Tests
~~~~~

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in
the `tests <https://github.com/iiconocimiento/aisee/tree/main/tests>`_ folder.


To run the tests you can run the following commands:

.. code-block:: console

    pip install poetry
    poetry install --with dev
    poetry run pytest

We use `ruff <https://beta.ruff.rs/docs/>`_ to run the code style checks, in order to run them, execute the following commmands:

.. code-block:: console

    pip install poetry
    poetry install --with dev
    poetry run ruff .

When you create a PR, the github workflow tests will also run.
